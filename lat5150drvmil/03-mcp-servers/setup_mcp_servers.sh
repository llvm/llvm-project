#!/bin/bash

# MCP Servers Automated Setup Script
# This script installs all MCP servers using the recommended method for each

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/user/LAT5150DRVMIL/03-mcp-servers"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  MCP Servers Automated Setup${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create base directory
print_info "Creating directory structure..."
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"
print_success "Directory structure created at $BASE_DIR"
echo ""

# ============================================================================
# Prerequisites Check
# ============================================================================

print_info "Checking prerequisites..."
MISSING_DEPS=()

# Check Python
if ! command_exists python3; then
    MISSING_DEPS+=("python3")
else
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
fi

# Check Node.js
if ! command_exists node; then
    MISSING_DEPS+=("node")
else
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION found"
fi

# Check Docker
if ! command_exists docker; then
    print_warning "Docker not found (optional for mcp-for-security)"
else
    DOCKER_VERSION=$(docker --version)
    print_success "$DOCKER_VERSION found"
fi

# Check Git
if ! command_exists git; then
    MISSING_DEPS+=("git")
else
    print_success "Git found"
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    print_error "Missing required dependencies: ${MISSING_DEPS[*]}"
    echo "Please install missing dependencies and run this script again."
    exit 1
fi

echo ""

# ============================================================================
# 1. Install search-tools-mcp
# ============================================================================

print_info "Installing search-tools-mcp..."

if [ -d "search-tools-mcp" ]; then
    print_warning "search-tools-mcp directory already exists, skipping clone"
else
    git clone https://github.com/voxmenthe/search-tools-mcp.git
    print_success "Cloned search-tools-mcp repository"
fi

cd search-tools-mcp

# Check for uv
if ! command_exists uv; then
    print_info "Installing uv package manager..."
    pip3 install uv
    print_success "uv installed"
fi

# Check for ripgrep
if ! command_exists rg; then
    print_warning "ripgrep (rg) not found. Please install it manually:"
    echo "  - Ubuntu/Debian: sudo apt-get install ripgrep"
    echo "  - macOS: brew install ripgrep"
    echo "  - Other: https://github.com/BurntSushi/ripgrep#installation"
else
    print_success "ripgrep found"
fi

# Sync dependencies
print_info "Running uv sync..."
uv sync
print_success "search-tools-mcp dependencies synced"

cd "$BASE_DIR"
echo ""

# ============================================================================
# 2. Install MetasploitMCP
# ============================================================================

print_info "Installing MetasploitMCP..."

if [ -d "MetasploitMCP" ]; then
    print_warning "MetasploitMCP directory already exists, skipping clone"
else
    git clone https://github.com/GH05TCREW/MetasploitMCP.git
    print_success "Cloned MetasploitMCP repository"
fi

cd MetasploitMCP

if [ -f "requirements.txt" ]; then
    print_info "Installing Python dependencies..."
    pip3 install -r requirements.txt
    print_success "MetasploitMCP dependencies installed"
else
    print_warning "requirements.txt not found in MetasploitMCP"
fi

print_warning "MetasploitMCP requires Metasploit Framework to be installed separately"
print_info "You'll need to run: msfrpcd -P your_password -S -a 127.0.0.1 -p 55553"
print_info "Don't forget to update MSF_PASSWORD in mcp_servers_config.json"

cd "$BASE_DIR"
echo ""

# ============================================================================
# 3. Install mcp-for-security
# ============================================================================

print_info "Installing mcp-for-security..."

# Try Docker first (recommended)
if command_exists docker; then
    print_info "Docker found - pulling mcp-for-security Docker image..."
    if docker pull cyprox/mcp-for-security 2>/dev/null; then
        print_success "mcp-for-security Docker image pulled successfully"
        print_info "You can also use the git installation below as an alternative"
    else
        print_warning "Docker pull failed, falling back to git installation"
    fi
fi

# Git installation (always do this as it's used in the config)
if [ -d "mcp-for-security" ]; then
    print_warning "mcp-for-security directory already exists, skipping clone"
else
    git clone https://github.com/cyproxio/mcp-for-security.git
    print_success "Cloned mcp-for-security repository"
fi

cd mcp-for-security

if [ -f "start.sh" ]; then
    print_info "Making start.sh executable..."
    chmod +x start.sh
    print_success "start.sh is now executable"
    print_warning "Individual security tools may need separate installation"
    print_info "Run 'bash start.sh' to see which tools need to be installed"
else
    print_warning "start.sh not found in mcp-for-security"
fi

cd "$BASE_DIR"
echo ""

# ============================================================================
# 4. Setup maigret (npx - no installation needed)
# ============================================================================

print_info "Setting up maigret..."

# Create reports directory
mkdir -p maigret-reports
print_success "Created maigret-reports directory"

# Check Docker for maigret
if ! command_exists docker; then
    print_warning "Docker not found - maigret requires Docker to run"
    print_info "Install Docker Desktop (macOS/Windows) or Docker Engine (Linux)"
else
    print_success "Docker is ready for maigret"
fi

print_info "maigret will auto-download via npx when first run"

echo ""

# ============================================================================
# 5. docs-mcp-server (npx - no installation needed)
# ============================================================================

print_info "Setting up docs-mcp-server..."
print_info "docs-mcp-server will auto-download via npx when first run"
print_info "Optional: Set OPENAI_API_KEY for enhanced semantic search"
echo ""

# ============================================================================
# 6. CVE Scraper - Automated Telegram CVE Monitoring
# ============================================================================

print_info "Setting up Telegram CVE Scraper..."

RAG_DIR="/home/user/LAT5150DRVMIL/rag_system"
cd "$RAG_DIR"

# Install Python dependencies
print_info "Installing telethon and python-dotenv..."
pip3 install telethon python-dotenv
print_success "CVE scraper dependencies installed"

# Create .env.telegram if it doesn't exist
ENV_FILE="$RAG_DIR/.env.telegram"
if [ ! -f "$ENV_FILE" ]; then
    print_info "Creating Telegram configuration..."
    cat > "$ENV_FILE" <<'EOF'
# Telegram API Credentials
TELEGRAM_API_ID=37733572
TELEGRAM_API_HASH=5fbbca6becf772efa224be5af735ce66

# Channels to monitor (comma-separated)
# - cveNotify: CVE vulnerability notifications
# - secharvester: General security news
# - Pwn3rzs: Pwn/exploit news
# - androidMalware: Android malware reports
SECURITY_CHANNELS=cveNotify,secharvester,Pwn3rzs,androidMalware

# Auto-update settings
AUTO_UPDATE_EMBEDDINGS=true
UPDATE_BATCH_SIZE=10
UPDATE_INTERVAL_SECONDS=300
EOF
    print_success "Created .env.telegram configuration for 4 security channels"
else
    print_warning ".env.telegram already exists, skipping creation"
fi

# Make scraper executable
chmod +x "$RAG_DIR/telegram_cve_scraper.py"
print_success "CVE scraper is now executable"

# Prompt for systemd service installation
echo ""
print_info "CVE Scraper can run automatically via systemd timer"
echo "  - Runs on boot (1 minute delay)"
echo "  - Runs every 5 minutes"
echo "  - Daily full resync at 3 AM"
echo ""

if [ "$EUID" -eq 0 ]; then
    # Running as root, can install directly
    print_warning "Running as root - installing systemd service..."
    bash "$RAG_DIR/install_cve_service.sh"
    print_success "CVE scraper systemd service installed"
else
    # Not root, provide instructions
    read -p "Install CVE scraper as systemd service? (requires sudo) [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing systemd service..."
        sudo bash "$RAG_DIR/install_cve_service.sh"
        print_success "CVE scraper systemd service installed"
    else
        print_warning "Skipping systemd installation"
        print_info "You can install it later with: sudo bash $RAG_DIR/install_cve_service.sh"
    fi
fi

# First-time Telegram authentication
echo ""
print_info "First-time Telegram authentication required..."
print_warning "You'll need to authorize this app with your Telegram account"
echo ""

read -p "Authenticate with Telegram now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting Telegram authentication..."
    print_info "You'll receive a code on Telegram - enter it when prompted"
    echo ""

    # Run scraper in auth mode (will create session file)
    python3 "$RAG_DIR/telegram_cve_scraper.py" --stats || true

    if [ -f "$RAG_DIR/telegram_cve_session.session" ]; then
        print_success "Telegram authentication successful!"
        print_info "Running initial CVE sync..."
        python3 "$RAG_DIR/telegram_cve_scraper.py" --oneshot
    else
        print_warning "Authentication skipped or failed"
        print_info "You can authenticate later by running:"
        echo "  cd $RAG_DIR && python3 telegram_cve_scraper.py --stats"
    fi
else
    print_warning "Skipping Telegram authentication"
    print_info "You'll need to authenticate before CVE scraper can run:"
    echo "  cd $RAG_DIR && python3 telegram_cve_scraper.py --stats"
fi

cd "$BASE_DIR"
echo ""

# ============================================================================
# Verification
# ============================================================================

print_info "Verifying installation..."
echo ""

echo "Directory structure:"
ls -la "$BASE_DIR"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

print_success "Installed MCP Servers:"
echo "  ✓ search-tools-mcp (requires: uv, ripgrep, kit)"
echo "  ✓ MetasploitMCP (requires: Metasploit Framework + msfrpcd)"
echo "  ✓ mcp-for-security (23 security tools - install individually)"
echo "  ✓ maigret (requires: Docker, auto-downloads via npx)"
echo "  ✓ docs-mcp-server (auto-downloads via npx)"
echo "  ✓ CVE Scraper (automated Telegram CVE monitoring)"
echo ""

print_info "Next Steps:"
echo ""
echo "1. Install missing tools:"
if ! command_exists rg; then
    echo "   - ripgrep: sudo apt-get install ripgrep (or brew install ripgrep)"
fi
if ! command_exists docker; then
    echo "   - Docker: https://docs.docker.com/get-docker/"
fi
echo ""

echo "2. Configure Metasploit:"
echo "   - Install Metasploit Framework: https://www.metasploit.com/"
echo "   - Start msfrpcd: msfrpcd -P your_password -S -a 127.0.0.1 -p 55553"
echo "   - Update MSF_PASSWORD in: /home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json"
echo ""

echo "3. Install security tools for mcp-for-security:"
echo "   cd $BASE_DIR/mcp-for-security"
echo "   bash start.sh  # This will show which tools need installation"
echo ""

echo "4. Optional - Set up API keys:"
echo "   - OPENAI_API_KEY for docs-mcp-server vector search"
echo ""

echo "5. CVE Scraper Management:"
echo "   - Status:  sudo systemctl status cve-scraper.timer"
echo "   - Logs:    sudo journalctl -u cve-scraper -f"
echo "   - Stats:   cd /home/user/LAT5150DRVMIL/rag_system && python3 telegram_cve_scraper.py --stats"
echo ""

echo "6. Verify configuration:"
echo "   cat /home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json"
echo ""

print_success "Setup script completed!"
print_info "For detailed documentation, see: /home/user/LAT5150DRVMIL/02-ai-engine/MCP_SERVERS_SETUP.md"
echo ""

# ============================================================================
# Test suggestions
# ============================================================================

echo -e "${BLUE}Quick Tests:${NC}"
echo ""
echo "Test search-tools-mcp:"
echo "  cd $BASE_DIR/search-tools-mcp && uv run mcp dev main.py"
echo ""
echo "Test docs-mcp-server:"
echo "  npx @arabold/docs-mcp-server@latest"
echo ""
echo "Test maigret:"
echo "  npx mcp-maigret@latest"
echo ""
echo "Test MetasploitMCP (after starting msfrpcd):"
echo "  cd $BASE_DIR/MetasploitMCP && python3 MetasploitMCP.py --transport stdio"
echo ""
echo "Test mcp-for-security:"
echo "  cd $BASE_DIR/mcp-for-security && bash start.sh"
echo ""

echo -e "${GREEN}Happy hacking!${NC}"
