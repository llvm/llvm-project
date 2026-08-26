# CLI_CLANG_TOOL2.py

import typer
import requests
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
from typing import List

app = typer.Typer()

# ------------------ CONFIG ------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
REPO = "llvm/llvm-project"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ UTILS ------------------
def fetch_pr_diff(pr_number: int) -> str:
    url = f"https://api.github.com/repos/{REPO}/pulls/{pr_number}"
    diff_url = requests.get(url, headers=HEADERS).json().get("diff_url")
    diff = requests.get(diff_url, headers=HEADERS).text
    return diff

def fetch_pr_title(pr_number: int) -> str:
    url = f"https://api.github.com/repos/{REPO}/pulls/{pr_number}"
    return requests.get(url, headers=HEADERS).json().get("title", "")

def load_spec_sections(path: str) -> List[str]:
    return Path(path).read_text(encoding="utf-8").split("\n\n")

def index_spec(sections: List[str]):
    embeddings = MODEL.encode(sections)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, sections

def retrieve_spec(query: str, index, sections, embeddings, k=3):
    q_emb = MODEL.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [sections[i] for i in I[0]]

def extract_openmp_pragmas_from_diff(diff: str) -> List[str]:
    return [line.strip() for line in diff.splitlines() if "#pragma omp" in line]

def build_prompt(diff: str, spec_snippets: List[str], pragmas: List[str] = []):
    prompt = f"""
You are an assistant summarizing OpenMP Pull Requests. Given the code diff and relevant sections of the OpenMP specification, write a structured summary for the PR.

## Diff:
{diff}

## Relevant Spec Sections:
{''.join(spec_snippets)}

## OpenMP Directives Detected:
{''.join(pragmas)}

## Summary:
"""
    return prompt

def generate_summary(prompt):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for code reviewers."},
                {"role": "user", "content": prompt},
            ]
        }
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()

SECTION_LINKS = {
    "parallel": "https://www.openmp.org/spec-html/6.0/openmpsu35.html",
    "worksharing": "https://www.openmp.org/spec-html/6.0/openmpsu46.html",
}

def build_markdown_output(summary_text: str, spec_snippets: List[str]) -> str:
    linked_snippets = []
    for snippet in spec_snippets:
        for title, url in SECTION_LINKS.items():
            if title.lower() in snippet.lower():
                snippet += f"\n\n🔗 [Spec Link]({url})"
                break
        linked_snippets.append(snippet)

    return f"""## 📝 Pull Request Summary

{summary_text}

---

## 📚 Related OpenMP Spec Sections
{''.join(f"```text\n{snippet}\n```\n\n" for snippet in linked_snippets)}"""

# ------------------ COMMANDS ------------------
@app.command()
def describe(pr_number: int, spec_path: str = r"D:\Web Development\CD_LAB\openmp_6_0_sections.txt"):
    """Generate structured PR summary from GitHub PR number."""
    title = fetch_pr_title(pr_number)
    diff = fetch_pr_diff(pr_number)
    sections = load_spec_sections(spec_path)
    index, emb, sec = index_spec(sections)

    snippets = retrieve_spec(title, index, sec, emb)
    pragmas = extract_openmp_pragmas_from_diff(diff)
    prompt = build_prompt(diff, snippets, pragmas)
    summary = generate_summary(prompt)

    print("\n\n## Generated Markdown PR Description\n")
    print(build_markdown_output(summary, snippets))

@app.command()
def local(diff_path: str, query: str, spec_path: str = r"D:\Web Development\CD_LAB\openmp_6_0_sections.txt"):
    """Generate summary from a local diff file and query string."""
    diff = Path(diff_path).read_text()
    sections = load_spec_sections(spec_path)
    index, emb, sec = index_spec(sections)

    snippets = retrieve_spec(query, index, sec, emb)
    pragmas = extract_openmp_pragmas_from_diff(diff)
    prompt = build_prompt(diff, snippets, pragmas)
    summary = generate_summary(prompt)

    print("\n\n## Generated Markdown PR Description\n")
    print(build_markdown_output(summary, snippets))

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app()

# Package Installation Guide:
# 1. Create setup.py or pyproject.toml
# 2. Add console_scripts:
# entry_points={
#     'console_scripts': [
#         'omp-pr-summary = CLI_CLANG_TOOL2:app',
#     ],
# }
# 3. Run: pip install -e .
# 4. Use as: omp-pr-summary describe 144229