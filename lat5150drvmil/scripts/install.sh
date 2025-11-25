#!/bin/bash
################################################################################
# LAT5150 DRVMIL - Unified Installation & Integration Script
# One-command setup for complete tactical AI sub-engine
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Installation configuration
INSTALL_DIR="/opt/lat5150"
SERVICE_NAME="lat5150-tactical-ai"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
WORKSPACE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Banner
clear
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}        LAT5150 DRVMIL - TACTICAL AI SUB-ENGINE${NC}"
echo -e "${CYAN}        Unified Installation & Integration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo
echo -e "${GREEN}Installation Details:${NC}"
echo -e "  Installation Directory: ${INSTALL_DIR}"
echo -e "  Workspace: ${WORKSPACE_PATH}"
echo -e "  Service User: ${SERVICE_USER}"
echo -e "  SystemD Service: ${SERVICE_NAME}"
echo
echo -e "${YELLOW}This script will:${NC}"
echo -e "  ✓ Install all system dependencies"
echo -e "  ✓ Set up Python virtual environment"
echo -e "  ✓ Configure directory structure"
echo -e "  ✓ Initialize state databases"
echo -e "  ✓ Install SystemD service (auto-start)"
echo -e "  ✓ Discover capabilities and resources"
echo -e "  ✓ Run system validation"
echo
read -p "Continue with installation? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}[ERROR]${NC} This script must be run as root (use sudo)"
   exit 1
fi

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}STEP 1/8: Installing System Dependencies${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MGR="apt-get"
    echo -e "${GREEN}[✓]${NC} Detected Debian/Ubuntu system"
elif command -v yum &> /dev/null; then
    PKG_MGR="yum"
    echo -e "${GREEN}[✓]${NC} Detected RHEL/CentOS system"
elif command -v dnf &> /dev/null; then
    PKG_MGR="dnf"
    echo -e "${GREEN}[✓]${NC} Detected Fedora system"
else
    echo -e "${RED}[ERROR]${NC} Unsupported package manager"
    exit 1
fi

# Update package cache
echo -e "${YELLOW}[INFO]${NC} Updating package cache..."
$PKG_MGR update -qq || true

# Install dependencies
PACKAGES=(
    "python3"
    "python3-pip"
    "python3-venv"
    "git"
    "curl"
    "jq"
    "sqlite3"
    "gcc"
    "make"
    "linux-headers-$(uname -r)"
)

# Add Docker/Podman if not present
if ! command -v docker &> /dev/null && ! command -v podman &> /dev/null; then
    if [[ "$PKG_MGR" == "apt-get" ]]; then
        PACKAGES+=("docker.io")
    else
        PACKAGES+=("docker")
    fi
fi

echo -e "${YELLOW}[INFO]${NC} Installing packages: ${PACKAGES[*]}"

for pkg in "${PACKAGES[@]}"; do
    if $PKG_MGR install -y "$pkg" &> /dev/null; then
        echo -e "${GREEN}[✓]${NC} $pkg"
    else
        echo -e "${YELLOW}[SKIP]${NC} $pkg (may already be installed)"
    fi
done

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}STEP 2/8: Setting Up Directory Structure${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Create directories
DIRECTORIES=(
    "$INSTALL_DIR"
    "$INSTALL_DIR/bin"
    "$INSTALL_DIR/config"
    "$INSTALL_DIR/state"
    "$INSTALL_DIR/logs"
    "$INSTALL_DIR/artifacts"
    "$INSTALL_DIR/audit"
    "$INSTALL_DIR/venv"
)

for dir in "${DIRECTORIES[@]}"; do
    if mkdir -p "$dir" 2>/dev/null; then
        echo -e "${GREEN}[✓]${NC} Created $dir"
    else
        echo -e "${YELLOW}[EXISTS]${NC} $dir"
    fi
done

# Set permissions
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
echo -e "${GREEN}[✓]${NC} Set permissions for $SERVICE_USER"

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}STEP 3/8: Setting Up Python Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Create virtual environment
if [ ! -d "$INSTALL_DIR/venv" ] || [ ! -f "$INSTALL_DIR/venv/bin/activate" ]; then
    echo -e "${YELLOW}[INFO]${NC} Creating Python virtual environment..."
    sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
    echo -e "${GREEN}[✓]${NC} Virtual environment created"
else
    echo -e "${YELLOW}[EXISTS]${NC} Virtual environment already exists"
fi

# Install Python packages
echo -e "${YELLOW}[INFO]${NC} Installing Python dependencies..."

PYTHON_PACKAGES=(
    "flask"
    "flask-cors"
    "aiohttp"
    "psutil"
    "anthropic"
    # MANDATORY AI Stack
    "openai>=1.0.0"
    "google-generativeai>=0.8.0"
    "pydantic>=2.9.0"
    "pydantic-ai>=0.0.13"
    "pydantic-settings>=2.5.0"
    "ollama>=0.3.0"
    "duckduckgo-search>=6.3.0"
    "shodan>=1.31.0"
    "httpx>=0.27.0"
    "fastapi>=0.115.0"
    "uvicorn[standard]>=0.30.0"
    # Utilities
    "requests>=2.31.0"
    "beautifulsoup4>=4.12.0"
    "pyyaml>=6.0"
    "python-dotenv>=1.0.0"
    "click>=8.1.0"
)

for pkg in "${PYTHON_PACKAGES[@]}"; do
    if sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -q "$pkg" 2>/dev/null; then
        echo -e "${GREEN}[✓]${NC} $pkg"
    else
        echo -e "${YELLOW}[SKIP]${NC} $pkg"
    fi
done

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}STEP 4/8: Installing Ollama (Local Models)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}[INFO]${NC} Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo -e "${GREEN}[✓]${NC} Ollama installed"
else
    echo -e "${YELLOW}[EXISTS]${NC} Ollama already installed"
fi

# Start Ollama service
if systemctl is-active --quiet ollama; then
    echo -e "${GREEN}[✓]${NC} Ollama service running"
else
    echo -e "${YELLOW}[INFO]${NC} Starting Ollama service..."
    systemctl enable ollama 2>/dev/null || true
    systemctl start ollama 2>/dev/null || true
    sleep 2
fi

# Pull default models
echo -e "${YELLOW}[INFO]${NC} Checking for local models..."
REQUIRED_MODELS=("whiterabbit-neo-33b" "qwen2.5-coder:7b")

for model in "${REQUIRED_MODELS[@]}"; do
    if ollama list 2>/dev/null | grep -q "${model%%:*}"; then
        echo -e "${GREEN}[✓]${NC} Model available: $model"
    else
        echo -e "${YELLOW}[INFO]${NC} Pulling model: $model..."
        echo -e "${YELLOW}[NOTE]${NC} This may take several minutes..."
        ollama pull "$model" || echo -e "${YELLOW}[WARN]${NC} Failed to pull $model"
    fi
done

# Pull legacy models (optional)
if ! ollama list 2>/dev/null | grep -q "deepseek-r1"; then
    echo -e "${YELLOW}[INFO]${NC} Pulling legacy model (deepseek-r1:1.5b)..."
    ollama pull deepseek-r1:1.5b || true
fi

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}STEP 5/8: Creating Master Entrypoint${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Create master entrypoint script
cat > "$INSTALL_DIR/bin/lat5150" <<'ENTRYPOINT_EOF'
#!/bin/bash
################################################################################
# LAT5150 DRVMIL - Master Entrypoint
################################################################################

INSTALL_DIR="/opt/lat5150"
WORKSPACE_PATH="__WORKSPACE_PATH__"

# Activate virtual environment
source "$INSTALL_DIR/venv/bin/activate"

# Set environment
export WORKSPACE_PATH="$WORKSPACE_PATH"
export OLLAMA_ENDPOINT="http://localhost:11434"
export PYTHONPATH="$WORKSPACE_PATH:$PYTHONPATH"

# Change to workspace
cd "$WORKSPACE_PATH/03-web-interface"

# Execute unified tactical API
exec python3 unified_tactical_api.py \
    --workspace "$WORKSPACE_PATH" \
    --ollama-endpoint "$OLLAMA_ENDPOINT" \
    --port 5001 \
    --host 0.0.0.0 \
    --local-models "whiterabbit-neo-33b,qwen2.5-coder:7b,deepseek-r1:1.5b"
ENTRYPOINT_EOF

# Replace placeholder
sed -i "s|__WORKSPACE_PATH__|$WORKSPACE_PATH|g" "$INSTALL_DIR/bin/lat5150"

# Make executable
chmod +x "$INSTALL_DIR/bin/lat5150"
chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/bin/lat5150"
echo -e "${GREEN}[✓]${NC} Master entrypoint created: $INSTALL_DIR/bin/lat5150"

# Create symlink
ln -sf "$INSTALL_DIR/bin/lat5150" /usr/local/bin/lat5150 2>/dev/null || true
echo -e "${GREEN}[✓]${NC} Symlink created: /usr/local/bin/lat5150"

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}STEP 6/8: Installing SystemD Service${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Create SystemD service
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<SERVICE_EOF
[Unit]
Description=LAT5150 DRVMIL Tactical AI Sub-Engine
Documentation=https://github.com/SWORDIntel/LAT5150DRVMIL
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$WORKSPACE_PATH
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=$WORKSPACE_PATH"
Environment="WORKSPACE_PATH=$WORKSPACE_PATH"
Environment="OLLAMA_ENDPOINT=http://localhost:11434"

ExecStartPre=/bin/sleep 5
ExecStart=$INSTALL_DIR/bin/lat5150

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=lat5150

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$INSTALL_DIR $WORKSPACE_PATH

[Install]
WantedBy=multi-user.target
SERVICE_EOF

echo -e "${GREEN}[✓]${NC} SystemD service created"

# Reload SystemD
systemctl daemon-reload
echo -e "${GREEN}[✓]${NC} SystemD daemon reloaded"

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}STEP 7/8: Initializing Self-Awareness System${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Run capability discovery
echo -e "${YELLOW}[INFO]${NC} Running capability discovery..."
cd "$WORKSPACE_PATH/03-web-interface"
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/python3" self_awareness_engine.py > /dev/null 2>&1 || true
echo -e "${GREEN}[✓]${NC} Capability discovery complete"

# Initialize state database
if [ -f "$INSTALL_DIR/state/self_awareness.db" ]; then
    echo -e "${GREEN}[✓]${NC} State database initialized"
else
    echo -e "${YELLOW}[WARN]${NC} State database not created"
fi

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}STEP 8/8: Running System Validation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Validation checklist
VALIDATION_PASSED=true

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}[✓]${NC} Python: $PYTHON_VERSION"
else
    echo -e "${RED}[✗]${NC} Python: NOT FOUND"
    VALIDATION_PASSED=false
fi

# Check Ollama
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}[✓]${NC} Ollama: Installed"
else
    echo -e "${YELLOW}[!]${NC} Ollama: Not installed (optional)"
fi

# Check Docker/Podman
if command -v docker &> /dev/null; then
    echo -e "${GREEN}[✓]${NC} Container Runtime: Docker"
elif command -v podman &> /dev/null; then
    echo -e "${GREEN}[✓]${NC} Container Runtime: Podman"
else
    echo -e "${YELLOW}[!]${NC} Container Runtime: Not found (optional for agents)"
fi

# Check directories
for dir in "$INSTALL_DIR" "$INSTALL_DIR/state" "$INSTALL_DIR/logs"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}[✓]${NC} Directory: $dir"
    else
        echo -e "${RED}[✗]${NC} Directory: $dir (MISSING)"
        VALIDATION_PASSED=false
    fi
done

# Check entrypoint
if [ -x "$INSTALL_DIR/bin/lat5150" ]; then
    echo -e "${GREEN}[✓]${NC} Entrypoint: $INSTALL_DIR/bin/lat5150"
else
    echo -e "${RED}[✗]${NC} Entrypoint: NOT EXECUTABLE"
    VALIDATION_PASSED=false
fi

# Check SystemD service
if systemctl list-unit-files | grep -q "$SERVICE_NAME"; then
    echo -e "${GREEN}[✓]${NC} SystemD Service: Installed"
else
    echo -e "${RED}[✗]${NC} SystemD Service: NOT INSTALLED"
    VALIDATION_PASSED=false
fi

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}✓ INSTALLATION SUCCESSFUL${NC}"
else
    echo -e "${YELLOW}⚠ INSTALLATION COMPLETED WITH WARNINGS${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

echo -e "${CYAN}Next Steps:${NC}"
echo
echo -e "${GREEN}1. Start the service:${NC}"
echo -e "   ${YELLOW}sudo systemctl start $SERVICE_NAME${NC}"
echo
echo -e "${GREEN}2. Enable auto-start on boot:${NC}"
echo -e "   ${YELLOW}sudo systemctl enable $SERVICE_NAME${NC}"
echo
echo -e "${GREEN}3. Check service status:${NC}"
echo -e "   ${YELLOW}sudo systemctl status $SERVICE_NAME${NC}"
echo
echo -e "${GREEN}4. View logs:${NC}"
echo -e "   ${YELLOW}sudo journalctl -u $SERVICE_NAME -f${NC}"
echo
echo -e "${GREEN}5. Access tactical UI:${NC}"
echo -e "   ${YELLOW}http://$(hostname -I | awk '{print $1}'):5001${NC}"
echo -e "   ${YELLOW}http://localhost:5001${NC}"
echo
echo -e "${GREEN}6. Manual start (for testing):${NC}"
echo -e "   ${YELLOW}lat5150${NC}"
echo
echo -e "${CYAN}Documentation:${NC}"
echo -e "  • Quick Start: $WORKSPACE_PATH/QUICKSTART_NL_INTEGRATION.md"
echo -e "  • Full Guide: $WORKSPACE_PATH/NATURAL_LANGUAGE_INTEGRATION.md"
echo -e "  • Deployment: $WORKSPACE_PATH/DEPLOYMENT_GUIDE.md"
echo
echo -e "${CYAN}Configuration:${NC}"
echo -e "  • Config Dir: $INSTALL_DIR/config"
echo -e "  • State DB: $INSTALL_DIR/state/self_awareness.db"
echo -e "  • Logs: $INSTALL_DIR/logs"
echo -e "  • Artifacts: $INSTALL_DIR/artifacts"
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Installation complete! The LAT5150 DRVMIL Tactical AI${NC}"
echo -e "${GREEN}Sub-Engine is ready for deployment.${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo
