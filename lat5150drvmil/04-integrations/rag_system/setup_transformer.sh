#!/bin/bash
#
# LAT5150DRVMIL RAG - Transformer Upgrade Setup
# Phase 1: Upgrade from TF-IDF (51.8%) to Transformers (75-88%)
#
# Requirements:
#   - 1.5GB free disk space
#   - Python 3.8+
#   - Internet connection (first run only)
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  LAT5150DRVMIL RAG - Transformer Upgrade Setup"
echo "════════════════════════════════════════════════════════════"
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check disk space
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$available_space" -lt 2 ]; then
    echo "⚠ WARNING: Low disk space (${available_space}GB). Need at least 2GB."
    read -p "Continue anyway? [y/N]: " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "✓ Available space: ${available_space}GB"
echo

# Step 1: Install dependencies
echo "────────────────────────────────────────────────────────────"
echo "Step 1: Installing transformer dependencies"
echo "────────────────────────────────────────────────────────────"
echo
echo "This will install:"
echo "  • sentence-transformers (HuggingFace embeddings)"
echo "  • transformers (core library)"
echo "  • torch (PyTorch - CPU version)"
echo
echo "Total download: ~1.5GB (one-time)"
echo

read -p "Continue with installation? [Y/n]: " install_confirm
if [[ $install_confirm =~ ^[Nn]$ ]]; then
    echo "Skipping installation. Install manually with:"
    echo "  pip install sentence-transformers transformers torch"
    echo
else
    echo "Installing dependencies..."
    pip install --no-cache-dir sentence-transformers transformers torch
    echo
    echo "✓ Dependencies installed"
    echo
fi

# Step 2: Check if index exists
echo "────────────────────────────────────────────────────────────"
echo "Step 2: Checking document index"
echo "────────────────────────────────────────────────────────────"
echo

if [ ! -f "processed_docs.json" ]; then
    echo "⚠ Document index not found!"
    echo "Building index from documentation..."
    echo
    python3 document_processor.py
    echo
fi

echo "✓ Document index ready"
echo

# Step 3: Generate transformer embeddings
echo "────────────────────────────────────────────────────────────"
echo "Step 3: Generating semantic embeddings"
echo "────────────────────────────────────────────────────────────"
echo
echo "This will:"
echo "  1. Download BAAI/bge-base-en-v1.5 model (~400MB, first run)"
echo "  2. Generate embeddings for all document chunks"
echo "  3. Save embeddings for fast loading"
echo
echo "Time: ~5-10 minutes (first run), ~10 seconds (subsequent)"
echo

read -p "Generate embeddings now? [Y/n]: " embed_confirm
if [[ ! $embed_confirm =~ ^[Nn]$ ]]; then
    python3 transformer_upgrade.py
    echo
    echo "✓ Embeddings generated and saved"
    echo
else
    echo "Skipped. Run manually with:"
    echo "  python3 transformer_upgrade.py"
    echo
fi

# Step 4: Test accuracy
echo "────────────────────────────────────────────────────────────"
echo "Step 4: Testing upgraded system"
echo "────────────────────────────────────────────────────────────"
echo

if [ -f "transformer_embeddings.npz" ]; then
    read -p "Run accuracy tests? [Y/n]: " test_confirm
    if [[ ! $test_confirm =~ ^[Nn]$ ]]; then
        echo
        python3 test_transformer_rag.py
    else
        echo "Skipped. Run manually with:"
        echo "  python3 test_transformer_rag.py"
    fi
else
    echo "⚠ Embeddings not found. Skipping tests."
    echo "Run transformer_upgrade.py first."
fi

echo
echo "════════════════════════════════════════════════════════════"
echo "  Transformer Upgrade Complete!"
echo "════════════════════════════════════════════════════════════"
echo
echo "Usage:"
echo "  • Single query:     python3 transformer_query.py \"your question\""
echo "  • Interactive:      python3 transformer_query.py"
echo "  • Compare results:  python3 transformer_query.py --compare \"question\""
echo
echo "Expected performance:"
echo "  • Accuracy:     75-88% (up from 51.8%)"
echo "  • Response:     0.5-1.5s (cached embeddings)"
echo
echo "Next steps:"
echo "  • Run setup_peft.sh for fine-tuning (90-95%+ accuracy)"
echo "  • Integrate with LLM (Ollama + Llama3)"
echo
