import gradio as gr
from model_utils import *


callback = gr.CSVLogger()
with gr.Blocks() as demo:
    #gr.HTML("<img src='/home/ywang_radformation_com/gradio_RAG/radformation-logo-white.png'") 
    #gr.Image("/file=/home/ywang_radformation_com/gradio_RAG/radformation-logo-white.png")
    github_banner_path = 'https://radformation.com/images/radformation-logo-white.svg'
    gr.HTML(f'<p align="center"><a href="https://radformation.com/"><img src={github_banner_path} width="700"/></a></p>')
    gr.Markdown('''# Retrieval Augmented Generation \n
        RAG involves creating a knowledge base containing two types of knowledge: parametric knowledge from LLM training and source knowledge from external input. Data for the knowledge base is derived from datasets relevant to the use case, which are then processed into smaller chunks to enhance relevance and efficiency. A vector database could be used to manage and search through the embeddings efficiently.''')
    with gr.Row():
        with gr.Column(variant = 'panel'):
            gr.Markdown("## Upload Document & Select the Embedding Model")
            file = gr.File(type="filepath")
            with gr.Row(equal_height=True):
                
                with gr.Column(variant = 'panel'):
                    embedding_model = gr.Dropdown(choices= ["Llama-2-7b-chat-hf","bge-large-en-v1.5","all-roberta-large-v1_1024d", "all-mpnet-base-v2_768d"],
                                    value="Llama-2-7b-chat-hf",
                                    label= "Select the embedding model")

                with gr.Column(variant='compact'):
                    vector_index_btn = gr.Button('Create vector store', variant='primary',scale=1)
                    vector_index_msg_out = gr.Textbox(show_label=False, lines=1,scale=1, placeholder="Creating vectore store ...")

            instruction = gr.Textbox(label="System instruction", lines=3, value="Use the following pieces of context to answer the question at the end by.You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.")
            reset_inst_btn = gr.Button('Reset',variant='primary', size = 'sm')

            with gr.Accordion(label="Text generation tuning parameters"):
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.1, step=0.05)
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=1, maximum=2048, value=512, step=1)
                repetition_penalty = gr.Slider(label="repetition_penalty", minimum=0, maximum=2, value=1.1, step=0.1)
                top_k= gr.Slider(label="top_k", minimum=1, maximum=1000, value=10, step=1)
                top_p=gr.Slider(label="top_p", minimum=0, maximum=1, value=0.95, step=0.05)
                k_context=gr.Slider(label="k_context", minimum=1, maximum=15, value=5, step=1)

            vector_index_btn.click(upload_and_create_vector_store,[file,embedding_model],vector_index_msg_out)
            reset_inst_btn.click(reset_sys_instruction,instruction,instruction)

        with gr.Column(variant = 'panel'):
            gr.Markdown("## Select the Generation Model")

            with gr.Row(equal_height=True):

                with gr.Column():
                    llm = gr.Dropdown(choices= ["Llamav2-7B-Chat", "Falcon-7B-Instruct"], value="Llamav2-7B-Chat", label="Select the LLM")
                    hf_token = gr.Textbox(label='Enter your valid HF token_id', type = "password")

                with gr.Column():
                    model_load_btn = gr.Button('Load model', variant='primary',scale=2)
                    load_success_msg = gr.Textbox(show_label=False,lines=1, placeholder="Model loading ...")
            chatbot = gr.Chatbot([], elem_id="chatbot",
                                label='Chatbox', height=725, )
            source_documents = gr.Textbox(label="Reference", lines=10)
            txt = gr.Textbox(label= "Question",lines=2,placeholder="Enter your question and press shift+enter ")
            
            with gr.Row():

                with gr.Column():
                    submit_btn = gr.Button('Submit',variant='primary', size = 'sm')

                with gr.Column():
                    clear_btn = gr.Button('Clear',variant='stop',size = 'sm')

            with gr.Row():
                with gr.Column():
                    feedback = gr.Text(label="Feedback",lines=10,placeholder="Enter your feedback and press shift+enter")
                    flag = gr.Button("Flag")
            # This needs to be called at some point prior to the first call to callback.flag()
            callback.setup([txt,chatbot,flag], "flagged_data_points")

            model_load_btn.click(load_models, [hf_token,embedding_model,llm], load_success_msg, api_name="load_models")

            txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot,instruction,temperature,max_new_tokens,repetition_penalty,top_k,top_p,k_context], [chatbot,source_documents])
            submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot,instruction,temperature, max_new_tokens,repetition_penalty,top_k,top_p,k_context], [chatbot,source_documents]).then(
                    clear_cuda_cache, None, None
                )


            clear_btn.click(lambda: ([],'',''), None, [chatbot,txt,source_documents], queue=False)
            feedback.submit(lambda: None,feedback,None)
            flag.click(lambda *args: callback.flag(args),[txt,chatbot,feedback],None,)



if __name__ == '__main__':
    #demo.queue(concurrency_count=3)
    #demo.launch(debug=True, share=True)
    demo.queue().launch(
            share=False,
            inbrowser=True,
            server_name='0.0.0.0',
            auth=("radllms", "X5L!2Nm7uwAnh7g")
            )
