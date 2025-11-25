#!/bin/bash
# DSMIL AI Engine Startup Script
# Validates all dependencies and starts services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           DSMIL AI Engine Startup Validator                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is NOT installed"
        return 1
    fi
}

# Function to check Python module
check_python_module() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python module '$1' is available"
        return 0
    else
        echo -e "${RED}✗${NC} Python module '$1' is NOT available"
        return 1
    fi
}

# Function to check service
check_service() {
    if curl -s "$1" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Service at $1 is running"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Service at $1 is NOT running"
        return 1
    fi
}

ALL_OK=true

echo -e "${BLUE}[1/5]${NC} Checking system dependencies..."
check_command python3 || ALL_OK=false
check_command curl || ALL_OK=false
check_command git || ALL_OK=false
echo ""

echo -e "${BLUE}[2/5]${NC} Checking Python modules..."
check_python_module requests || ALL_OK=false
check_python_module json || ALL_OK=false
check_python_module pathlib || ALL_OK=false
echo ""

echo -e "${BLUE}[3/5]${NC} Checking Ollama service..."
if ! check_service "http://localhost:11434/api/tags"; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Ollama is not running or not installed${NC}"
    echo -e ""
    echo -e "To install Ollama:"
    echo -e "  ${GREEN}curl -fsSL https://ollama.com/install.sh | sh${NC}"
    echo -e ""
    echo -e "To start Ollama:"
    echo -e "  ${GREEN}ollama serve${NC}   (run in separate terminal)"
    echo -e ""
    echo -e "To pull models:"
    echo -e "  ${GREEN}ollama pull deepseek-r1:1.5b${NC}         (fast, 1.5GB)"
    echo -e "  ${GREEN}ollama pull deepseek-coder:6.7b-instruct${NC}  (code, 4GB)"
    echo -e "  ${GREEN}ollama pull qwen2.5-coder:7b${NC}        (quality, 4.5GB)"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    ALL_OK=false
fi
echo ""

echo -e "${BLUE}[4/5]${NC} Checking DSMIL AI Engine status..."
cd "$(dirname "$0")"
python3 dsmil_ai_engine.py status > /tmp/ai_status.json 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} DSMIL AI Engine is operational"

    # Check models
    MODELS_AVAILABLE=$(grep -c '"available": true' /tmp/ai_status.json 2>/dev/null || echo "0")
    echo -e "  Models available: $MODELS_AVAILABLE/4"

    if [ "$MODELS_AVAILABLE" = "0" ]; then
        echo -e "${YELLOW}  ⚠ No models available - please pull models (see above)${NC}"
        ALL_OK=false
    fi
else
    echo -e "${RED}✗${NC} DSMIL AI Engine failed to start"
    cat /tmp/ai_status.json
    ALL_OK=false
fi
echo ""

echo -e "${BLUE}[5/7]${NC} Checking Intel GPU and vLLM..."
if [ -f "start_vllm_server.sh" ]; then
    echo -e "${GREEN}✓${NC} vLLM startup script found"
    # Check if vLLM server is already running
    if ! check_service "http://localhost:8000/health"; then
        echo -e "${YELLOW}  Starting vLLM server in background...${NC}"
        cd "$(dirname "$0")"
        tmux new-session -d -s vllm_server "./start_vllm_server.sh" 2>/dev/null || {
            echo -e "${YELLOW}  ⚠ tmux not available, starting in background${NC}"
            nohup ./start_vllm_server.sh > /tmp/vllm_server.log 2>&1 &
        }
        sleep 3
        echo -e "${GREEN}  vLLM server started (check tmux session 'vllm_server' or /tmp/vllm_server.log)${NC}"
    else
        echo -e "${GREEN}  vLLM server already running${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} vLLM startup script not found (run setup_ai_enhancements.sh)"
fi
echo ""

echo -e "${BLUE}[6/7]${NC} Checking MCP Servers..."
MCP_COUNT=0
if [ -f "$HOME/.config/mcp_servers_config.json" ]; then
    MCP_COUNT=$(grep -c "\"command\"" "$HOME/.config/mcp_servers_config.json" 2>/dev/null || echo "0")
    echo -e "${GREEN}✓${NC} MCP configuration found ($MCP_COUNT servers configured)"
else
    echo -e "${YELLOW}⚠${NC} MCP configuration not found (run setup_mcp_servers.sh)"
fi
echo ""

echo -e "${BLUE}[7/7]${NC} Checking Web Server..."
cd ../03-web-interface
if [ -f "dsmil_unified_server.py" ]; then
    echo -e "${GREEN}✓${NC} Unified server found"
else
    echo -e "${RED}✗${NC} Unified server NOT found"
    ALL_OK=false
fi
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e ""
    echo -e "${GREEN}Starting DSMIL Unified Server...${NC}"
    echo -e "  Server will be available at: ${BLUE}http://localhost:9876${NC}"
    echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop"
    echo -e ""
    cd ../03-web-interface
    python3 dsmil_unified_server.py
else
    echo -e "${RED}✗ Some checks failed - please fix issues above${NC}"
    echo -e ""
    echo -e "Common fixes:"
    echo -e "  1. Install Ollama: ${GREEN}curl -fsSL https://ollama.com/install.sh | sh${NC}"
    echo -e "  2. Start Ollama: ${GREEN}ollama serve${NC}"
    echo -e "  3. Pull models: ${GREEN}ollama pull deepseek-r1:1.5b${NC}"
    echo -e "  4. Install Python deps: ${GREEN}pip3 install requests${NC}"
    exit 1
fi
