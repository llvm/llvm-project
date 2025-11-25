#!/bin/bash
#
# Local Deployment Script for LAT5150DRVMIL AI Engine
# Deploys to local development environment
#

set -euo pipefail

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${DEPLOY_DIR:-${HOME}/.local/share/lat5150drvmil}"
CONFIG_DIR="${CONFIG_DIR:-${HOME}/.config/lat5150drvmil}"
LOG_DIR="${LOG_DIR:-${HOME}/.local/share/lat5150drvmil/logs}"
DATA_DIR="${DATA_DIR:-${HOME}/.local/share/lat5150drvmil/data}"
BACKUP_DIR="${BACKUP_DIR:-${HOME}/.local/share/lat5150drvmil/backups}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  LAT5150DRVMIL - Local Deployment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Validate environment
echo -e "${BLUE}[1/8] Validating environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"

# Step 2: Create directories
echo -e "${BLUE}[2/8] Creating directories...${NC}"
mkdir -p "$DEPLOY_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}✓ Directories created${NC}"

# Step 3: Backup existing installation
echo -e "${BLUE}[3/8] Creating backup...${NC}"
if [ -d "$DEPLOY_DIR/02-ai-engine" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/backup_${TIMESTAMP}.tar.gz"
    tar -czf "$BACKUP_FILE" -C "$DEPLOY_DIR" . 2>/dev/null || true
    echo -e "${GREEN}✓ Backup created: $BACKUP_FILE${NC}"
else
    echo -e "${YELLOW}⊙ No existing installation to backup${NC}"
fi

# Step 4: Copy files
echo -e "${BLUE}[4/8] Copying files...${NC}"
rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='build' \
    --exclude='dist' \
    "$PROJECT_ROOT/02-ai-engine" \
    "$PROJECT_ROOT/03-mcp-servers" \
    "$DEPLOY_DIR/"
echo -e "${GREEN}✓ Files copied${NC}"

# Step 5: Install/Update dependencies
echo -e "${BLUE}[5/8] Installing dependencies...${NC}"
cd "$PROJECT_ROOT"
python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1
python3 -m pip install -r requirements.txt >/dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 6: Copy/Update configuration
echo -e "${BLUE}[6/8] Setting up configuration...${NC}"
if [ ! -f "$CONFIG_DIR/config.json" ]; then
    if [ -f "$PROJECT_ROOT/config/config.example.json" ]; then
        cp "$PROJECT_ROOT/config/config.example.json" "$CONFIG_DIR/config.json"
        echo -e "${YELLOW}⊙ Created default configuration (please review: $CONFIG_DIR/config.json)${NC}"
    fi
fi

# Copy MCP server config
if [ -f "$PROJECT_ROOT/02-ai-engine/mcp_servers_config.json" ]; then
    cp "$PROJECT_ROOT/02-ai-engine/mcp_servers_config.json" "$CONFIG_DIR/"
    echo -e "${GREEN}✓ MCP configuration updated${NC}"
fi

# Step 7: Set up environment variables
echo -e "${BLUE}[7/8] Setting up environment...${NC}"
ENV_FILE="$CONFIG_DIR/env.sh"
cat > "$ENV_FILE" <<EOF
# LAT5150DRVMIL Environment Variables
export LAT5150_HOME="$DEPLOY_DIR"
export LAT5150_CONFIG="$CONFIG_DIR"
export LAT5150_LOGS="$LOG_DIR"
export LAT5150_DATA="$DATA_DIR"
export PYTHONPATH="\${LAT5150_HOME}/02-ai-engine:\${PYTHONPATH:-}"
EOF
chmod +x "$ENV_FILE"
echo -e "${GREEN}✓ Environment configured${NC}"
echo -e "${YELLOW}⊙ Source this file to activate: source $ENV_FILE${NC}"

# Step 8: Validate installation
echo -e "${BLUE}[8/8] Validating installation...${NC}"
export PYTHONPATH="$DEPLOY_DIR/02-ai-engine:${PYTHONPATH:-}"

# Check if modules can be imported
if python3 -c "import sys; sys.path.insert(0, '$DEPLOY_DIR/02-ai-engine'); from unified_orchestrator import UnifiedOrchestrator" 2>/dev/null; then
    echo -e "${GREEN}✓ Module imports successful${NC}"
else
    echo -e "${YELLOW}⊙ Module imports failed (may need configuration)${NC}"
fi

# Check config files
if [ -f "$CONFIG_DIR/config.json" ]; then
    python3 -c "import json; json.load(open('$CONFIG_DIR/config.json'))" 2>/dev/null && \
        echo -e "${GREEN}✓ Configuration valid${NC}" || \
        echo -e "${RED}✗ Configuration invalid${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Local Deployment Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Deployment Details:${NC}"
echo -e "  Installation: ${DEPLOY_DIR}"
echo -e "  Configuration: ${CONFIG_DIR}"
echo -e "  Logs: ${LOG_DIR}"
echo -e "  Data: ${DATA_DIR}"
echo -e "  Backups: ${BACKUP_DIR}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Source environment: ${GREEN}source $ENV_FILE${NC}"
echo -e "  2. Review config: ${GREEN}$CONFIG_DIR/config.json${NC}"
echo -e "  3. Start MCP servers: ${GREEN}make mcp-start${NC}"
echo -e "  4. Run tests: ${GREEN}make test${NC}"
echo ""
