#!/bin/bash
################################################################################
# LAT5150 DRVMIL - Tactical AI Sub-Engine Launcher
# Starts unified tactical API with LOCAL-FIRST natural language interface
################################################################################

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}LAT5150 DRVMIL - TACTICAL AI SUB-ENGINE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
WORKSPACE_PATH="${WORKSPACE_PATH:-/home/user/LAT5150DRVMIL}"
OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"
API_PORT="${API_PORT:-5001}"

# Local models (WhiteRabbit as default)
LOCAL_MODELS="${LOCAL_MODELS:-whiterabbit,llama3.2:latest,codellama:latest,mixtral:latest}"

echo -e "${YELLOW}[CONFIG]${NC} Workspace: ${WORKSPACE_PATH}"
echo -e "${YELLOW}[CONFIG]${NC} Ollama Endpoint: ${OLLAMA_ENDPOINT}"
echo -e "${YELLOW}[CONFIG]${NC} API Port: ${API_PORT}"
echo -e "${YELLOW}[CONFIG]${NC} Local Models: ${LOCAL_MODELS}"
echo

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python3 not found"
    exit 1
fi

echo -e "${GREEN}[CHECK]${NC} Python3 available: $(python3 --version)"

# Check Ollama availability
echo -e "${YELLOW}[CHECK]${NC} Checking Ollama connectivity..."
if curl -s -f "${OLLAMA_ENDPOINT}/api/version" > /dev/null 2>&1; then
    OLLAMA_VERSION=$(curl -s "${OLLAMA_ENDPOINT}/api/version" | jq -r '.version' 2>/dev/null || echo "unknown")
    echo -e "${GREEN}[CHECK]${NC} Ollama available: ${OLLAMA_VERSION}"
else
    echo -e "${YELLOW}[WARN]${NC} Ollama not responding at ${OLLAMA_ENDPOINT}"
    echo -e "${YELLOW}[WARN]${NC} NL processing will fall back to rule-based mode"
fi

# Check if WhiteRabbit model is available
echo -e "${YELLOW}[CHECK]${NC} Checking for WhiteRabbit model..."
if curl -s "${OLLAMA_ENDPOINT}/api/tags" 2>/dev/null | grep -q "whiterabbit"; then
    echo -e "${GREEN}[CHECK]${NC} WhiteRabbit model available"
else
    echo -e "${YELLOW}[WARN]${NC} WhiteRabbit model not found in Ollama"
    echo -e "${YELLOW}[INFO]${NC} To use WhiteRabbit, ensure it's loaded in Ollama"
fi

# Check required Python packages
echo -e "${YELLOW}[CHECK]${NC} Checking Python dependencies..."
REQUIRED_PACKAGES=("flask" "flask_cors" "aiohttp")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}[WARN]${NC} Missing packages: ${MISSING_PACKAGES[*]}"
    echo -e "${YELLOW}[INFO]${NC} Installing missing packages..."
    pip3 install -q "${MISSING_PACKAGES[@]}"
fi

echo -e "${GREEN}[CHECK]${NC} All Python dependencies available"
echo

# Check integration files exist
echo -e "${YELLOW}[CHECK]${NC} Verifying integration files..."
REQUIRED_FILES=(
    "capability_registry.py"
    "natural_language_processor.py"
    "unified_tactical_api.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}[ERROR]${NC} Missing file: $file"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} $file"
done

echo

# Launch unified tactical API
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[LAUNCH]${NC} Starting Unified Tactical API"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo
echo -e "${GREEN}[INFO]${NC} API will be available at: http://127.0.0.1:${API_PORT}"
echo -e "${GREEN}[INFO]${NC} Tactical UI: http://127.0.0.1:${API_PORT}/"
echo -e "${GREEN}[INFO]${NC} Natural Language API: http://127.0.0.1:${API_PORT}/api/v2/nl/command"
echo -e "${GREEN}[INFO]${NC} Self-Awareness: http://127.0.0.1:${API_PORT}/api/v2/self-awareness"
echo
echo -e "${YELLOW}[INFO]${NC} Press Ctrl+C to stop"
echo

# Export environment variables
export WORKSPACE_PATH
export OLLAMA_ENDPOINT
export LOCAL_MODELS

# Launch the API
exec python3 unified_tactical_api.py \
    --workspace "$WORKSPACE_PATH" \
    --ollama-endpoint "$OLLAMA_ENDPOINT" \
    --port "$API_PORT" \
    --local-models "$LOCAL_MODELS"
