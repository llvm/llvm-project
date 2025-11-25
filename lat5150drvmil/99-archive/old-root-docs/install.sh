#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# DSMIL Unified AI Platform - Installation Script
# Version: 8.3
# Dell Latitude 5450 Covert Edition - LOCAL-FIRST AI
#═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Installation directory
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_HOME="$HOME"

#═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
#═══════════════════════════════════════════════════════════════════════════════

print_banner() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  DSMIL UNIFIED AI PLATFORM v8.3 - INSTALLER"
    echo "  Dell Latitude 5450 Covert Edition"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "Do not run this script as root. Run as normal user."
        print_warning "The script will ask for sudo when needed."
        exit 1
    fi
}

#═══════════════════════════════════════════════════════════════════════════════
# Dependency Checking
#═══════════════════════════════════════════════════════════════════════════════

check_dependencies() {
    print_step "Checking dependencies..."

    local missing=()

    # Check for required commands
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    command -v pip3 >/dev/null 2>&1 || missing+=("python3-pip")
    command -v git >/dev/null 2>&1 || missing+=("git")
    command -v curl >/dev/null 2>&1 || missing+=("curl")

    if [ ${#missing[@]} -ne 0 ]; then
        print_warning "Missing dependencies: ${missing[*]}"
        print_step "Installing missing dependencies..."
        sudo apt update
        sudo apt install -y python3 python3-pip git curl wget
    fi

    print_success "All system dependencies satisfied"
}

#═══════════════════════════════════════════════════════════════════════════════
# Python Environment Setup
#═══════════════════════════════════════════════════════════════════════════════

setup_python_env() {
    print_step "Setting up Python environment..."

    # Install Python packages
    pip3 install --user --upgrade pip

    print_step "Installing Python dependencies..."
    pip3 install --user \
        requests \
        anthropic \
        google-generativeai \
        openai \
        flask \
        flask-cors \
        beautifulsoup4 \
        lxml \
        numpy \
        sentence-transformers \
        faiss-cpu

    print_success "Python environment configured"
}

#═══════════════════════════════════════════════════════════════════════════════
# Ollama Installation
#═══════════════════════════════════════════════════════════════════════════════

install_ollama() {
    print_step "Checking Ollama installation..."

    if command -v ollama >/dev/null 2>&1; then
        print_success "Ollama already installed"
    else
        print_step "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        print_success "Ollama installed"
    fi

    # Start Ollama service
    sudo systemctl enable ollama 2>/dev/null || true
    sudo systemctl start ollama 2>/dev/null || true

    print_step "Waiting for Ollama to start..."
    sleep 3
}

#═══════════════════════════════════════════════════════════════════════════════
# Model Download
#═══════════════════════════════════════════════════════════════════════════════

download_models() {
    print_step "Downloading AI models..."

    echo -e "${YELLOW}This may take 10-30 minutes depending on your connection...${NC}"

    # Essential models
    if ! ollama list | grep -q "deepseek-r1:1.5b"; then
        print_step "Downloading DeepSeek R1 1.5B (fast reasoning)..."
        ollama pull deepseek-r1:1.5b
    else
        print_success "DeepSeek R1 already downloaded"
    fi

    if ! ollama list | grep -q "qwen2.5-coder:1.5b"; then
        print_step "Downloading Qwen Coder 1.5B (code specialist)..."
        ollama pull qwen2.5-coder:1.5b
    else
        print_success "Qwen Coder already downloaded"
    fi

    # Optional: Ask about larger models
    echo -e "\n${CYAN}Optional: Download larger models for better quality?${NC}"
    echo "  - deepseek-coder:6.7b (better code generation, ~4GB)"
    echo "  - codellama:7b (Meta's code model, ~4GB)"
    read -p "Download optional models? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ollama pull deepseek-coder:6.7b
        ollama pull codellama:7b
    fi

    print_success "Models ready"
}

#═══════════════════════════════════════════════════════════════════════════════
# Systemd Service Setup
#═══════════════════════════════════════════════════════════════════════════════

setup_systemd_service() {
    print_step "Setting up DSMIL service..."

    # Check if service file exists
    if [ ! -f "$INSTALL_DIR/05-deployment/systemd/dsmil-server.service" ]; then
        print_warning "Service file not found, creating..."
        mkdir -p "$INSTALL_DIR/05-deployment/systemd"

        cat > "$INSTALL_DIR/05-deployment/systemd/dsmil-server.service" << EOF
[Unit]
Description=DSMIL Unified AI Server
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR/03-web-interface
ExecStart=/usr/bin/python3 $INSTALL_DIR/03-web-interface/dsmil_unified_server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF
    fi

    # Install service
    sudo cp "$INSTALL_DIR/05-deployment/systemd/dsmil-server.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable dsmil-server.service

    print_success "Systemd service installed"
}

#═══════════════════════════════════════════════════════════════════════════════
# DSMIL Framework Setup
#═══════════════════════════════════════════════════════════════════════════════

setup_dsmil_framework() {
    print_step "Configuring DSMIL framework..."

    # Check if DSMIL module exists
    if [ -d "$INSTALL_DIR/01-source" ]; then
        print_success "DSMIL framework source found"
    else
        print_warning "DSMIL framework source not found"
        print_warning "Platform will work but without hardware attestation"
    fi

    # Create logs directory
    mkdir -p "$INSTALL_DIR/logs"

    # Create RAG index directory
    mkdir -p "$USER_HOME/.local/share/dsmil/rag_index"

    print_success "DSMIL framework configured"
}

#═══════════════════════════════════════════════════════════════════════════════
# Configuration
#═══════════════════════════════════════════════════════════════════════════════

configure_environment() {
    print_step "Configuring environment..."

    # Create config directory
    mkdir -p "$USER_HOME/.config/dsmil"

    # Create basic config file
    cat > "$USER_HOME/.config/dsmil/config.json" << EOF
{
    "version": "8.3",
    "install_dir": "$INSTALL_DIR",
    "local_models": {
        "reasoning": "deepseek-r1:1.5b",
        "code": "qwen2.5-coder:1.5b"
    },
    "server": {
        "host": "127.0.0.1",
        "port": 9876
    },
    "rag": {
        "index_dir": "$USER_HOME/.local/share/dsmil/rag_index"
    }
}
EOF

    print_success "Configuration saved to ~/.config/dsmil/config.json"
}

#═══════════════════════════════════════════════════════════════════════════════
# Verification
#═══════════════════════════════════════════════════════════════════════════════

verify_installation() {
    print_step "Verifying installation..."

    local errors=0

    # Check Python packages
    python3 -c "import requests, flask" 2>/dev/null || {
        print_error "Python packages not installed correctly"
        ((errors++))
    }

    # Check Ollama
    if ! command -v ollama >/dev/null 2>&1; then
        print_error "Ollama not found"
        ((errors++))
    fi

    # Check models
    if ! ollama list | grep -q "deepseek-r1:1.5b"; then
        print_error "DeepSeek R1 model not found"
        ((errors++))
    fi

    # Check service file
    if [ ! -f "/etc/systemd/system/dsmil-server.service" ]; then
        print_error "Systemd service not installed"
        ((errors++))
    fi

    if [ $errors -eq 0 ]; then
        print_success "All checks passed!"
    else
        print_error "Installation incomplete ($errors errors)"
        return 1
    fi
}

#═══════════════════════════════════════════════════════════════════════════════
# Start Service
#═══════════════════════════════════════════════════════════════════════════════

start_service() {
    print_step "Starting DSMIL service..."

    sudo systemctl start dsmil-server.service

    sleep 2

    if sudo systemctl is-active --quiet dsmil-server.service; then
        print_success "Service started successfully"
    else
        print_error "Service failed to start"
        print_warning "Check logs with: sudo journalctl -u dsmil-server.service -f"
        return 1
    fi
}

#═══════════════════════════════════════════════════════════════════════════════
# Final Instructions
#═══════════════════════════════════════════════════════════════════════════════

print_final_instructions() {
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  INSTALLATION COMPLETE!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}\n"

    echo -e "${CYAN}🚀 Access the interface:${NC}"
    echo -e "   Web UI: ${YELLOW}http://localhost:9876${NC}"
    echo -e "   Or run: ${YELLOW}xdg-open http://localhost:9876${NC}\n"

    echo -e "${CYAN}📋 Service Management:${NC}"
    echo -e "   Start:   ${YELLOW}sudo systemctl start dsmil-server${NC}"
    echo -e "   Stop:    ${YELLOW}sudo systemctl stop dsmil-server${NC}"
    echo -e "   Status:  ${YELLOW}sudo systemctl status dsmil-server${NC}"
    echo -e "   Logs:    ${YELLOW}sudo journalctl -u dsmil-server -f${NC}\n"

    echo -e "${CYAN}🔧 Manual Start (development):${NC}"
    echo -e "   ${YELLOW}cd $INSTALL_DIR/03-web-interface${NC}"
    echo -e "   ${YELLOW}python3 dsmil_unified_server.py${NC}\n"

    echo -e "${CYAN}📚 Features Available:${NC}"
    echo -e "   ✓ Local AI inference (DeepSeek R1 + Qwen Coder)"
    echo -e "   ✓ Auto-coding tools (Edit, Create, Debug, Refactor, Review, Tests, Docs)"
    echo -e "   ✓ Web search & crawling"
    echo -e "   ✓ RAG knowledge base"
    echo -e "   ✓ Smart routing"
    echo -e "   ✓ Hardware attestation (if DSMIL framework available)\n"

    echo -e "${CYAN}📖 Documentation:${NC}"
    echo -e "   README:  ${YELLOW}$INSTALL_DIR/README.md${NC}"
    echo -e "   Docs:    ${YELLOW}$INSTALL_DIR/00-documentation/${NC}"
    echo -e "   GitHub:  ${YELLOW}https://github.com/SWORDIntel/LAT5150DRVMIL${NC}\n"

    echo -e "${GREEN}Happy coding! 🎯${NC}\n"
}

#═══════════════════════════════════════════════════════════════════════════════
# Main Installation Flow
#═══════════════════════════════════════════════════════════════════════════════

main() {
    print_banner
    check_root

    echo -e "${YELLOW}This will install DSMIL Unified AI Platform v8.3${NC}"
    echo -e "${YELLOW}Installation directory: $INSTALL_DIR${NC}\n"

    read -p "Continue with installation? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_warning "Installation cancelled"
        exit 0
    fi

    # Run installation steps
    check_dependencies
    setup_python_env
    install_ollama
    download_models
    setup_dsmil_framework
    configure_environment
    setup_systemd_service

    # Verify
    if verify_installation; then
        # Ask to start service
        echo -e "\n${CYAN}Start DSMIL service now?${NC}"
        read -p "[Y/n]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            start_service
        fi

        print_final_instructions
    else
        print_error "Installation verification failed"
        print_warning "Please check errors above and try again"
        exit 1
    fi
}

# Run main installation
main "$@"
