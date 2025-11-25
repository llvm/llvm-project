#!/bin/bash
# DSMIL AI Engine - Uncensored Model Setup Script
# Installs recommended uncensored coding models with proper quantization

set -e

echo "=========================================="
echo "DSMIL AI - Uncensored Model Setup"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}✗ Ollama is not running${NC}"
    echo "Start Ollama with: systemctl start ollama"
    echo "Or: ollama serve"
    exit 1
fi

echo -e "${GREEN}✓ Ollama is running${NC}"
echo ""

# Detect available memory
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))

echo -e "${BLUE}System Memory: ${TOTAL_RAM_GB}GB${NC}"
echo ""

# Recommend models based on available memory
if [ $TOTAL_RAM_GB -lt 16 ]; then
    echo -e "${YELLOW}⚠ Low memory detected (${TOTAL_RAM_GB}GB)${NC}"
    echo "Recommended: Lightweight models only"
    RECOMMENDED_MODELS="mistral:7b-instruct-uncensored yi-coder:9b-chat-q4_K_M"
elif [ $TOTAL_RAM_GB -lt 32 ]; then
    echo -e "${YELLOW}⚠ Moderate memory (${TOTAL_RAM_GB}GB)${NC}"
    echo "Recommended: 34B models with Q4_K_M quantization"
    RECOMMENDED_MODELS="wizardlm-uncensored-codellama:34b-q4_K_M mistral:7b-instruct-uncensored yi-coder:9b-chat-q4_K_M"
else
    echo -e "${GREEN}✓ High memory (${TOTAL_RAM_GB}GB)${NC}"
    echo "Recommended: All models available"
    RECOMMENDED_MODELS="wizardlm-uncensored-codellama:34b-q4_K_M wizardcoder:34b-python-q4_K_M phind-codellama:34b-v2-q4_K_M mistral:7b-instruct-uncensored yi-coder:9b-chat-q4_K_M"
fi

echo ""
echo "=========================================="
echo "Installation Options"
echo "=========================================="
echo ""
echo "1) Essential only (Default model)"
echo "2) Recommended (Based on system memory)"
echo "3) Full suite (All uncensored models)"
echo "4) Custom selection"
echo "5) Exit"
echo ""
read -p "Choose [1-5]: " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Installing essential model...${NC}"
        MODELS="wizardlm-uncensored-codellama:34b-q4_K_M"
        ;;
    2)
        echo ""
        echo -e "${BLUE}Installing recommended models...${NC}"
        MODELS=$RECOMMENDED_MODELS
        ;;
    3)
        echo ""
        echo -e "${BLUE}Installing full suite...${NC}"
        MODELS="wizardlm-uncensored-codellama:34b-q4_K_M wizardcoder:34b-python-q4_K_M phind-codellama:34b-v2-q4_K_M deepseek-coder:33b-instruct-q4_K_M mistral:7b-instruct-uncensored yi-coder:9b-chat-q4_K_M codellama:34b-instruct-q4_K_M"
        ;;
    4)
        echo ""
        echo "Available models:"
        echo "  wizardlm-uncensored-codellama:34b-q4_K_M     (Default, 34B)"
        echo "  wizardcoder:34b-python-q4_K_M                (Python, 34B)"
        echo "  phind-codellama:34b-v2-q4_K_M                (Explanations, 34B)"
        echo "  deepseek-coder:33b-instruct-q4_K_M           (Multi-lang, 33B)"
        echo "  mistral:7b-instruct-uncensored               (Fast, 7B)"
        echo "  yi-coder:9b-chat-q4_K_M                      (Modern, 9B)"
        echo "  codellama:34b-instruct-q4_K_M                (General, 34B)"
        echo ""
        read -p "Enter model names (space-separated): " MODELS
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Installing Models"
echo "=========================================="
echo ""

INSTALLED=0
FAILED=0

for model in $MODELS; do
    echo -e "${BLUE}→ Pulling ${model}...${NC}"
    if ollama pull $model; then
        echo -e "${GREEN}✓ ${model} installed${NC}"
        ((INSTALLED++))
    else
        echo -e "${RED}✗ ${model} failed${NC}"
        ((FAILED++))
    fi
    echo ""
done

echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo -e "${GREEN}✓ Installed: ${INSTALLED}${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}✗ Failed: ${FAILED}${NC}"
fi
echo ""

# Test default model
echo "=========================================="
echo "Testing Default Model"
echo "=========================================="
echo ""

DEFAULT_MODEL="wizardlm-uncensored-codellama:34b-q4_K_M"

if ollama list | grep -q "$DEFAULT_MODEL"; then
    echo -e "${GREEN}✓ Default model ready: ${DEFAULT_MODEL}${NC}"
    echo ""
    echo "Testing inference..."
    echo ""

    # Quick test
    RESPONSE=$(ollama run $DEFAULT_MODEL "Write a one-line Python function to reverse a string" 2>&1 | head -n 5)
    echo "$RESPONSE"
    echo ""
    echo -e "${GREEN}✓ Model test successful${NC}"
else
    echo -e "${YELLOW}⚠ Default model not installed${NC}"
    echo "Install with: ollama pull $DEFAULT_MODEL"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Launch AI interface:"
echo "   cd /home/user/LAT5150DRVMIL/02-ai-engine"
echo "   python3 ai-tui-default"
echo ""
echo "2. Quick query:"
echo "   python3 ai.py \"Your question\""
echo ""
echo "3. Check model configuration:"
echo "   cat MODEL_CONFIG.md"
echo ""
echo "4. View device settings:"
echo "   python3 configure_device.py --status"
echo ""
echo -e "${GREEN}✓ Setup complete${NC}"
