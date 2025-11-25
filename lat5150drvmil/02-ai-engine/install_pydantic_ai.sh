#!/bin/bash
#
# Pydantic AI Installation Script
# Installs dependencies and tests the integration
#

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         DSMIL AI Engine - Pydantic AI Installation                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/5]${NC} Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo -e "${RED}Error: Python $required_version or higher required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python $python_version"
echo ""

# Check if pip is available
echo -e "${YELLOW}[2/5]${NC} Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 not found${NC}"
    echo "Install with: sudo apt install python3-pip"
    exit 1
fi
echo -e "${GREEN}✓${NC} pip3 available"
echo ""

# Install Pydantic AI dependencies
echo -e "${YELLOW}[3/5]${NC} Installing Pydantic AI dependencies..."
echo "This may take a minute..."
pip3 install -r requirements-pydantic-ai.txt --quiet --upgrade

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${RED}✗${NC} Installation failed"
    echo "Try: pip3 install -r requirements-pydantic-ai.txt"
    exit 1
fi
echo ""

# Check Ollama
echo -e "${YELLOW}[4/5]${NC} Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Ollama service running"

    # Check for models
    models=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('models', [])))")
    echo -e "${GREEN}✓${NC} Found $models model(s) installed"
else
    echo -e "${YELLOW}⚠${NC}  Ollama not running or not installed"
    echo ""
    echo "To install Ollama:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  ollama serve &"
    echo "  ollama pull deepseek-r1:1.5b  # Fast model"
fi
echo ""

# Test Pydantic AI integration
echo -e "${YELLOW}[5/5]${NC} Testing Pydantic AI integration..."

# Test 1: Import test
python3 -c "from pydantic_models import DSMILQueryRequest, DSMILQueryResult; print('✓ Pydantic models imported')" 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} Failed to import Pydantic models"
    exit 1
fi

# Test 2: Engine creation test
python3 -c "
from dsmil_ai_engine import DSMILAIEngine, PYDANTIC_AVAILABLE
print(f'✓ AI Engine imported')
print(f'✓ Pydantic support: {PYDANTIC_AVAILABLE}')
engine = DSMILAIEngine(pydantic_mode=False)
print(f'✓ Engine created (legacy mode)')
engine_pyd = DSMILAIEngine(pydantic_mode=True)
print(f'✓ Engine created (Pydantic mode)')
" 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} Failed to create AI engine"
    exit 1
fi

# Test 3: Quick benchmark (if requested)
if [[ "$1" == "--benchmark" ]]; then
    echo ""
    echo -e "${YELLOW}Running performance benchmark...${NC}"
    python3 benchmark_binary_vs_pydantic.py
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    Installation Complete!                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Pydantic AI is ready to use!${NC}"
echo ""
echo "Quick Start:"
echo ""
echo "1. Legacy Mode (dict responses):"
echo "   from dsmil_ai_engine import DSMILAIEngine"
echo "   engine = DSMILAIEngine(pydantic_mode=False)"
echo "   result = engine.generate('Your query')"
echo "   print(result['response'])  # dict"
echo ""
echo "2. Pydantic Mode (type-safe):"
echo "   from dsmil_ai_engine import DSMILAIEngine"
echo "   from pydantic_models import DSMILQueryRequest"
echo "   engine = DSMILAIEngine(pydantic_mode=True)"
echo "   request = DSMILQueryRequest(prompt='Your query')"
echo "   result = engine.generate(request)"
echo "   print(result.response)  # Pydantic model with autocomplete!"
echo ""
echo "3. Read the docs:"
echo "   cat PYDANTIC_AI_INTEGRATION.md"
echo ""
echo "Run benchmark: ./install_pydantic_ai.sh --benchmark"
echo ""
