#!/bin/bash
# Setup script for RAG-Enhanced Code Assistant
# Installs Ollama and downloads coding models

set -e

echo "================================"
echo "Code Assistant Setup"
echo "================================"
echo

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama already installed"
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
fi

echo
echo "Available Coding Models:"
echo "1. deepseek-coder:6.7b  (Recommended - Best for code, 4GB)"
echo "2. codellama:7b         (Good balance, 4GB)"
echo "3. qwen2.5-coder:7b     (Multilingual, 4GB)"
echo "4. deepseek-coder:1.3b  (Smallest, 1GB - for low RAM)"
echo

read -p "Select model [1-4, default=1]: " choice
choice=${choice:-1}

case $choice in
    1)
        MODEL="deepseek-coder:6.7b"
        ;;
    2)
        MODEL="codellama:7b"
        ;;
    3)
        MODEL="qwen2.5-coder:7b"
        ;;
    4)
        MODEL="deepseek-coder:1.3b"
        ;;
    *)
        echo "Invalid choice, using deepseek-coder:6.7b"
        MODEL="deepseek-coder:6.7b"
        ;;
esac

echo
echo "Downloading $MODEL (this may take 5-10 minutes)..."
ollama pull "$MODEL"

echo
echo "✓ Setup complete!"
echo
echo "Usage:"
echo "  # Interactive mode"
echo "  python3 rag_system/code_assistant.py -i"
echo
echo "  # Ask a coding question"
echo "  python3 rag_system/code_assistant.py 'Write a function to parse CVE data'"
echo
echo "  # Generate code"
echo "  python3 rag_system/code_assistant.py '/code Create a kernel module loader'"
echo
echo "  # Make it executable"
echo "  chmod +x rag_system/code_assistant.py"
echo "  ./rag_system/code_assistant.py -i"
echo
