import requests
import json
import os
from pathlib import Path
import readline
import glob

class DeepSeekCodeAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"  # 请替换为实际API端点
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_code(self, code_file_path, prompt_template=None):
        # 读取代码文件
        with open(code_file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # 准备提示词
        if prompt_template is None:
            prompt = f"""
请分析以下C++代码的时间复杂度以及可以改进的点：

{code_content}

请提供:
1. 整体时间复杂度分析
2. 关键函数/方法的时间复杂度
3. 内存使用分析
4. 具体的改进建议
5. 代码优化示例
"""
        else:
            prompt = prompt_template.format(code=code_content)
        
        # 准备API请求数据
        payload = {
            "model": "deepseek-coder",  # 使用代码专用模型
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # 低随机性以获得更确定的回答
            "max_tokens": 4000   # 根据需要进行调整
        }
        
        # 发送请求
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=None  # None = 無限等待
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            analysis = result['choices'][0]['message']['content']
            
            return analysis
            
        except Exception as e:
            print(f"API请求失败: {e}")
            return None
    
    def save_analysis(self, analysis_result, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(analysis_result)
        print(f"分析结果已保存到: {output_path}")

def _complete_path(text, state):
    # 展開 ~ 與 $ENV
    text = os.path.expanduser(os.path.expandvars(text))

    # 空字串時，預設從當前目錄開始
    if not text:
        text = './'

    # 拆出目錄與前綴
    dirname, prefix = os.path.split(text)
    if not dirname:
        dirname = '.'

    try:
        entries = os.listdir(dirname)
    except Exception:
        entries = []

    # 做匹配
    matches = []
    for name in entries:
        if name.startswith(prefix):
            full = os.path.join(dirname, name)
            # 目錄補全時自動加 '/'
            if os.path.isdir(full):
                full += '/'
            matches.append(full)

    # 也支援使用者原本寫的 glob 型式（例如 src/*.cpp）
    if '*' in text or '?' in text or '[' in text:
        matches.extend(glob.glob(text))

    # 去重、排序
    matches = sorted(set(matches))

    return matches[state] if state < len(matches) else None

# 讓 '/' 不被當成分隔符，保留路徑連續性
readline.set_completer_delims(' \t\n;')

# macOS 內建 Python 多半是 libedit；Linux/自裝 Python 多半是 GNU readline
if 'libedit' in readline.__doc__:
    # libedit 的綁定語法
    readline.parse_and_bind("bind ^I rl_complete")
else:
    # GNU readline 的綁定語法
    readline.parse_and_bind("tab: complete")

readline.set_completer(_complete_path)

# 使用示例
if __name__ == "__main__":
    # 替换为您的API密钥
    API_KEY = "sk-9cafb2e074bf4b348af4a075c19ccf6b"
    
    # 创建分析器实例
    analyzer = DeepSeekCodeAnalyzer(API_KEY)
    
    # 讓使用者輸入檔案名稱
    code_file = input("請輸入要分析的程式碼檔案名稱（含路徑）：").strip()
    
    # 自定义提示词（可选）
    custom_prompt = """
作为资深C++性能优化专家，请详细分析以下LLVM IRTranslator代码：

{code}

请重点关注：
1. 算法复杂度分析（最好、最坏、平均情况）
2. 内存访问模式和缓存友好性
3. 潜在的性能瓶颈
4. 并行化机会
5. LLVM特定最佳实践
6. 具体重构建议和代码示例
"""
    
    # 执行分析
    print("正在分析代码，请稍候...")
    result = analyzer.analyze_code(code_file, custom_prompt)
    
    if result:
        # 保存分析结果
        output_file = "code_analysis_report.txt"
        analyzer.save_analysis(result, output_file)
        
        # 打印部分结果预览
        print("\n分析结果预览:")
        print("=" * 50)
        print(result)
    else:
        print("分析失败")
