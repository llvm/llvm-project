#!/bin/bash
#
# Complete AI Stack Setup - LAT5150DRVMIL
# Installs all mandatory dependencies and configures the system
#
# Usage: ./setup_complete_ai_stack.sh [--skip-heavy]
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_HEAVY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-heavy)
            SKIP_HEAVY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-heavy]"
            exit 1
            ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     LAT5150DRVMIL - Complete AI Stack Setup                          ║"
echo "║     Hardware-Attested, Type-Safe AI Inference                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check Python version
echo "[1/10] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "✓ Python $PYTHON_VERSION (meets requirement: ≥$REQUIRED_VERSION)"
else
    echo "✗ Python $PYTHON_VERSION does not meet requirement: ≥$REQUIRED_VERSION"
    exit 1
fi

# Step 2: Check pip
echo ""
echo "[2/10] Checking pip..."
if command -v pip3 &> /dev/null; then
    PIP_VERSION=$(pip3 --version | awk '{print $2}')
    echo "✓ pip $PIP_VERSION installed"
else
    echo "✗ pip3 not found. Install with: sudo apt install python3-pip"
    exit 1
fi

# Step 3: Install mandatory AI dependencies
echo ""
echo "[3/10] Installing mandatory AI dependencies..."
echo "This includes: openai, google-generativeai, pydantic, pydantic-ai, ollama, duckduckgo-search, shodan"

pip3 install --user \
    'openai>=1.0.0' \
    'google-generativeai>=0.8.0' \
    'pydantic>=2.9.0' \
    'pydantic-ai>=0.0.13' \
    'pydantic-settings>=2.5.0' \
    'ollama>=0.3.0' \
    'duckduckgo-search>=6.3.0' \
    'shodan>=1.31.0' \
    'httpx>=0.27.0' \
    'fastapi>=0.115.0' \
    'uvicorn[standard]>=0.30.0'

echo "✓ Mandatory AI dependencies installed"

# Step 4: Install lightweight dependencies
echo ""
echo "[4/10] Installing lightweight dependencies..."
pip3 install --user \
    'requests>=2.31.0' \
    'beautifulsoup4>=4.12.0' \
    'pyyaml>=6.0' \
    'python-dotenv>=1.0.0' \
    'click>=8.1.0' \
    'psutil>=5.9.0'

echo "✓ Lightweight dependencies installed"

# Step 5: Install heavy ML dependencies (optional)
if [ "$SKIP_HEAVY" = false ]; then
    echo ""
    echo "[5/10] Installing heavy ML dependencies..."
    echo "This may take 10-20 minutes depending on your connection..."

    pip3 install --user \
        'torch>=2.0.0' \
        'transformers>=4.30.0' \
        'sentence-transformers>=2.2.0' \
        'intel-extension-for-pytorch>=2.0.0' \
        'numpy>=1.24.0' \
        'scipy>=1.10.0' \
        'pandas>=2.0.0'

    echo "✓ Heavy ML dependencies installed"
else
    echo ""
    echo "[5/10] Skipping heavy ML dependencies (--skip-heavy flag set)"
fi

# Step 6: Check Ollama installation
echo ""
echo "[6/10] Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | head -n1)
    echo "✓ Ollama installed: $OLLAMA_VERSION"

    # Check if Ollama service is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✓ Ollama service is running"
    else
        echo "⚠  Ollama installed but service not running"
        echo "   Start with: ollama serve"
    fi
else
    echo "⚠  Ollama not installed"
    echo "   Install from: https://ollama.com/download"
    echo "   Or run: curl -fsSL https://ollama.com/install.sh | sh"
fi

# Step 7: Pull recommended Ollama models
echo ""
echo "[7/10] Checking Ollama models..."
if command -v ollama &> /dev/null; then
    MODELS_NEEDED=("whiterabbit-neo-33b" "qwen2.5-coder:7b" "deepseek-r1:1.5b")

    for model in "${MODELS_NEEDED[@]}"; do
        if ollama list 2>/dev/null | grep -q "^${model%%:*}"; then
            echo "✓ Model available: $model"
        else
            echo "⊘ Model not found: $model"
            echo "   Pull with: ollama pull $model"
        fi
    done
else
    echo "⊘ Skipping model check (Ollama not installed)"
fi

# Step 8: Set up API keys (interactive)
echo ""
echo "[8/10] Checking API keys..."

# Check OpenAI
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⊘ OPENAI_API_KEY not set"
    echo "   Get key from: https://platform.openai.com/api-keys"
    echo "   Set with: export OPENAI_API_KEY='your_key'"
    echo "   Add to ~/.bashrc for persistence"
else
    echo "✓ OPENAI_API_KEY is set"
fi

# Check Gemini
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⊘ GEMINI_API_KEY not set"
    echo "   Get key from: https://aistudio.google.com/apikey"
    echo "   Set with: export GEMINI_API_KEY='your_key'"
else
    echo "✓ GEMINI_API_KEY is set"
fi

# Check Shodan (optional)
if [ -z "$SHODAN_API_KEY" ]; then
    echo "⊘ SHODAN_API_KEY not set (optional)"
    echo "   Get key from: https://account.shodan.io/"
else
    echo "✓ SHODAN_API_KEY is set"
fi

# Step 9: Test Pydantic integration
echo ""
echo "[9/10] Testing Pydantic AI integration..."
cd "$SCRIPT_DIR/02-ai-engine"

if python3 test_imports.py 2>&1 | grep -q "All import and syntax tests passed"; then
    echo "✓ Pydantic AI integration tests passed"
else
    echo "⚠  Some integration tests failed (check logs above)"
fi

# Step 10: Summary
echo ""
echo "[10/10] Setup Summary"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "✓ Mandatory AI dependencies installed:"
echo "  - OpenAI API (structured outputs)"
echo "  - Google Gemini API (multimodal)"
echo "  - Pydantic + Pydantic AI (type safety)"
echo "  - Ollama client (local inference)"
echo "  - DuckDuckGo Search (web search)"
echo "  - Shodan (threat intelligence)"
echo ""

if [ "$SKIP_HEAVY" = false ]; then
    echo "✓ Heavy ML dependencies installed (PyTorch, Transformers, etc.)"
else
    echo "⊘ Heavy ML dependencies skipped (use --skip-heavy to include)"
fi

echo ""
echo "Next Steps:"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Set API keys (if not already set):"
echo "   export OPENAI_API_KEY='your_key'"
echo "   export GEMINI_API_KEY='your_key'"
echo "   export SHODAN_API_KEY='your_key'  # Optional"
echo ""
echo "2. Install Ollama models:"
echo "   ollama pull whiterabbit-neo-33b      # Primary model (NPU/GPU/NCS2)"
echo "   ollama pull qwen2.5-coder:7b         # Quality code generation"
echo "   ollama pull deepseek-r1:1.5b         # Legacy fast model"
echo ""
echo "3. Start AI server:"
echo "   cd 02-ai-engine"
echo "   ./start_ai_server.sh"
echo ""
echo "4. Test the system:"
echo "   # CLI (dict mode)"
echo "   python3 ai.py 'What is TPM attestation?'"
echo ""
echo "   # CLI (Pydantic type-safe mode)"
echo "   python3 ai.py --pydantic 'What is TPM attestation?'"
echo ""
echo "   # Web API"
echo "   curl 'http://localhost:9876/ai/chat?msg=hello&pydantic=1'"
echo ""
echo "5. Run comprehensive tests:"
echo "   cd 02-ai-engine"
echo "   python3 test_dual_mode.py  # Requires Ollama running"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Setup complete! The system is ready for type-safe AI inference."
echo "═══════════════════════════════════════════════════════════════════════"
