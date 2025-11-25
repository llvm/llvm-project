#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# DSMIL COMPLETE INSTALLATION - Framework + AI System
# Version: 8.3.2
# Dell Latitude 5450 Covert Edition - COMPLETE SETUP
#═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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
    echo "  DSMIL COMPLETE INSTALLATION v8.3.2"
    echo "  Framework + AI System + Hardware Attestation"
    echo "  Dell Latitude 5450 Covert Edition"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
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
# System Requirements Check
#═══════════════════════════════════════════════════════════════════════════════

check_system_requirements() {
    print_section "CHECKING SYSTEM REQUIREMENTS"

    # Check OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_step "OS: $PRETTY_NAME"
        if [[ "$ID" != "debian" && "$ID" != "ubuntu" ]]; then
            print_warning "This script is designed for Debian/Ubuntu. Proceed with caution."
        fi
    fi

    # Check kernel version
    KERNEL_VERSION=$(uname -r)
    print_step "Kernel: $KERNEL_VERSION"

    # Check architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]]; then
        print_error "This system requires x86_64 architecture"
        exit 1
    fi
    print_success "Architecture: $ARCH"

    # Check available RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -lt 8 ]; then
        print_warning "Less than 8GB RAM detected. System may be slow."
    else
        print_success "RAM: ${TOTAL_RAM}GB"
    fi

    # Check available disk space
    AVAILABLE_SPACE=$(df -BG "$INSTALL_DIR" | awk 'NR==2{print $4}' | tr -d 'G')
    if [ "$AVAILABLE_SPACE" -lt 20 ]; then
        print_error "Less than 20GB free space. Need at least 20GB."
        exit 1
    fi
    print_success "Free Space: ${AVAILABLE_SPACE}GB"
}

#═══════════════════════════════════════════════════════════════════════════════
# Dependency Installation
#═══════════════════════════════════════════════════════════════════════════════

install_system_dependencies() {
    print_section "INSTALLING SYSTEM DEPENDENCIES"

    local packages=(
        # Core build tools
        build-essential
        linux-headers-$(uname -r)
        dkms

        # Python environment
        python3
        python3-pip
        python3-venv
        python3-dev

        # Development tools
        git
        curl
        wget
        ca-certificates

        # AI/ML libraries
        libopenblas-dev
        liblapack-dev

        # Hardware tools
        pciutils
        usbutils
        dmidecode

        # Monitoring tools
        htop
        iotop
        sysstat
    )

    print_step "Updating package lists..."
    sudo apt update

    print_step "Installing ${#packages[@]} system packages..."
    sudo apt install -y "${packages[@]}" || {
        print_error "Some packages failed to install"
        print_warning "Continuing anyway..."
    }

    print_success "System dependencies installed"
}

install_python_dependencies() {
    print_section "INSTALLING PYTHON DEPENDENCIES"

    print_step "Upgrading pip..."
    pip3 install --user --upgrade pip setuptools wheel

    print_step "Installing AI/ML libraries..."
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
        scipy \
        pandas \
        sentence-transformers \
        faiss-cpu \
        transformers \
        torch \
        tokenizers

    print_success "Python dependencies installed"
}

#═══════════════════════════════════════════════════════════════════════════════
# DSMIL Framework Installation
#═══════════════════════════════════════════════════════════════════════════════

install_dsmil_framework() {
    print_section "INSTALLING DSMIL FRAMEWORK"

    if [ ! -d "$INSTALL_DIR/01-source" ]; then
        print_warning "DSMIL source not found. Skipping framework installation."
        return
    fi

    cd "$INSTALL_DIR/01-source"

    # Check if kernel module exists
    if [ -f "kernel/dsmil-72dev.ko" ]; then
        print_step "Installing DSMIL kernel module..."

        # Copy module to system
        sudo mkdir -p /lib/modules/$(uname -r)/extra/dsmil
        sudo cp kernel/dsmil-72dev.ko /lib/modules/$(uname -r)/extra/dsmil/

        # Update module dependencies
        sudo depmod -a

        # Load module
        if sudo modprobe dsmil-72dev 2>/dev/null; then
            print_success "DSMIL kernel module loaded"
        else
            print_warning "DSMIL module not loaded (may require specific hardware)"
        fi
    fi

    # Install userspace tools
    if [ -d "userspace-tools" ]; then
        print_step "Installing DSMIL userspace tools..."
        if [ -f "userspace-tools/Makefile" ]; then
            cd userspace-tools
            make clean 2>/dev/null || true
            if make; then
                sudo make install
                print_success "Userspace tools installed"
            else
                print_warning "Userspace tools build failed"
            fi
            cd ..
        fi
    fi

    cd "$INSTALL_DIR"
    print_success "DSMIL framework setup complete"
}

#═══════════════════════════════════════════════════════════════════════════════
# Ollama & Model Installation
#═══════════════════════════════════════════════════════════════════════════════

install_ollama() {
    print_section "INSTALLING OLLAMA AI RUNTIME"

    if command -v ollama >/dev/null 2>&1; then
        print_success "Ollama already installed"
    else
        print_step "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        print_success "Ollama installed"
    fi

    # Enable and start Ollama service
    sudo systemctl enable ollama 2>/dev/null || true
    sudo systemctl start ollama 2>/dev/null || true

    print_step "Waiting for Ollama to start..."
    sleep 3

    # Check Ollama status
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_success "Ollama is running"
    else
        print_warning "Ollama may not be running properly"
    fi
}

download_ai_models() {
    print_section "DOWNLOADING AI MODELS"

    echo -e "${YELLOW}This will download ~3GB of AI models. Continue? [Y/n]${NC}"
    read -r response
    if [[ "$response" =~ ^[Nn]$ ]]; then
        print_warning "Skipping model downloads"
        return
    fi

    # Essential models
    local models=(
        "deepseek-r1:1.5b"
        "qwen2.5-coder:1.5b"
    )

    for model in "${models[@]}"; do
        print_step "Downloading $model..."
        if ollama list | grep -q "$model"; then
            print_success "$model already downloaded"
        else
            if ollama pull "$model"; then
                print_success "$model downloaded"
            else
                print_warning "$model download failed"
            fi
        fi
    done

    # Optional larger models
    echo -e "\n${CYAN}Download larger models for better quality? (requires ~8GB more)${NC}"
    echo "  - deepseek-coder:6.7b (better code generation)"
    echo "  - codellama:7b (Meta's code model)"
    read -p "[y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ollama pull deepseek-coder:6.7b
        ollama pull codellama:7b
    fi

    print_success "Model downloads complete"
}

#═══════════════════════════════════════════════════════════════════════════════
# Hardware Optimization
#═══════════════════════════════════════════════════════════════════════════════

setup_hardware_optimization() {
    print_section "HARDWARE OPTIMIZATION"

    # Check for Intel NPU
    if lspci | grep -qi "neural"; then
        print_step "Intel NPU detected"
        print_success "NPU available for acceleration"
    fi

    # Check for Intel GPU
    if lspci | grep -qi "intel.*graphics"; then
        print_step "Intel GPU detected"
        print_success "GPU available for acceleration"
    fi

    # Check for NCS2
    if lsusb | grep -qi "movidius"; then
        print_step "Intel Neural Compute Stick 2 detected"
        print_success "NCS2 available for edge inference"
    fi

    # Enable huge pages for better performance
    if [ -f /proc/sys/vm/nr_hugepages ]; then
        print_step "Enabling huge pages for performance..."
        echo 128 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null
        print_success "Huge pages enabled"
    fi

    print_success "Hardware optimization complete"
}

#═══════════════════════════════════════════════════════════════════════════════
# Service Configuration
#═══════════════════════════════════════════════════════════════════════════════

setup_systemd_service() {
    print_section "CONFIGURING SYSTEMD SERVICE"

    # Create service file
    print_step "Creating DSMIL service..."

    sudo tee /etc/systemd/system/dsmil-server.service > /dev/null << EOF
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

# Environment
Environment="PATH=/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    sudo systemctl daemon-reload
    sudo systemctl enable dsmil-server.service

    print_success "Systemd service configured"
}

#═══════════════════════════════════════════════════════════════════════════════
# Configuration Files
#═══════════════════════════════════════════════════════════════════════════════

create_configuration() {
    print_section "CREATING CONFIGURATION"

    # Create config directories
    mkdir -p "$USER_HOME/.config/dsmil"
    mkdir -p "$USER_HOME/.local/share/dsmil/rag_index"
    mkdir -p "$INSTALL_DIR/logs"

    # Create main config
    cat > "$USER_HOME/.config/dsmil/config.json" << EOF
{
    "version": "8.3.2",
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
    },
    "hardware": {
        "npu_enabled": true,
        "gpu_enabled": true,
        "ncs2_enabled": true
    },
    "dsmil": {
        "mode": "STANDARD",
        "attestation_enabled": true
    }
}
EOF

    print_success "Configuration created"
}

#═══════════════════════════════════════════════════════════════════════════════
# Verification
#═══════════════════════════════════════════════════════════════════════════════

verify_installation() {
    print_section "VERIFYING INSTALLATION"

    local errors=0

    # Check Python packages
    print_step "Checking Python packages..."
    if python3 -c "import requests, flask, numpy, sentence_transformers" 2>/dev/null; then
        print_success "Python packages OK"
    else
        print_error "Python packages incomplete"
        ((errors++))
    fi

    # Check Ollama
    print_step "Checking Ollama..."
    if command -v ollama >/dev/null 2>&1; then
        print_success "Ollama installed"
    else
        print_error "Ollama not found"
        ((errors++))
    fi

    # Check models
    print_step "Checking AI models..."
    if ollama list | grep -q "deepseek-r1:1.5b"; then
        print_success "DeepSeek R1 model ready"
    else
        print_warning "DeepSeek R1 model not found"
    fi

    # Check service file
    print_step "Checking systemd service..."
    if [ -f "/etc/systemd/system/dsmil-server.service" ]; then
        print_success "Service file exists"
    else
        print_error "Service file not found"
        ((errors++))
    fi

    # Check DSMIL framework
    print_step "Checking DSMIL framework..."
    if lsmod | grep -q dsmil; then
        print_success "DSMIL kernel module loaded"
    else
        print_warning "DSMIL module not loaded (may be normal)"
    fi

    # SECURITY: Verify localhost-only binding
    print_step "Verifying security configuration..."
    if grep -q 'HOST = "127.0.0.1"' "$INSTALL_DIR/03-web-interface/dsmil_unified_server.py"; then
        print_success "Server configured for localhost-only (SECURE)"
    else
        print_error "Server NOT configured for localhost-only!"
        print_warning "SECURITY RISK: Server may be exposed to network"
        ((errors++))
    fi

    if [ $errors -eq 0 ]; then
        print_success "All checks passed!"
        return 0
    else
        print_error "Installation incomplete ($errors errors)"
        return 1
    fi
}

#═══════════════════════════════════════════════════════════════════════════════
# Start Service
#═══════════════════════════════════════════════════════════════════════════════

start_service() {
    print_section "STARTING DSMIL SERVICE"

    print_step "Starting service..."
    sudo systemctl start dsmil-server.service

    sleep 2

    if sudo systemctl is-active --quiet dsmil-server.service; then
        print_success "Service started successfully"

        # Test endpoint
        if curl -s http://localhost:9876/status >/dev/null 2>&1; then
            print_success "Web interface is accessible"
        fi

        # SECURITY: Verify localhost-only binding
        print_step "Verifying network security..."
        if sudo ss -tlnp | grep 9876 | grep -q "127.0.0.1:9876"; then
            print_success "Server bound to localhost only (SECURE ✓)"
        elif sudo ss -tlnp | grep 9876 | grep -q "0.0.0.0:9876"; then
            print_error "WARNING: Server exposed to network (0.0.0.0)!"
            print_error "This is a SECURITY RISK! Fix immediately."
            return 1
        fi
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

    echo -e "${CYAN}🔧 Configuration:${NC}"
    echo -e "   Config:  ${YELLOW}~/.config/dsmil/config.json${NC}"
    echo -e "   RAG:     ${YELLOW}~/.local/share/dsmil/rag_index/${NC}"
    echo -e "   Logs:    ${YELLOW}$INSTALL_DIR/logs/${NC}\n"

    echo -e "${CYAN}📚 Features Available:${NC}"
    echo -e "   ✓ 7 auto-coding tools (Edit, Create, Debug, Refactor, Review, Tests, Docs)"
    echo -e "   ✓ Chat history persistence (auto-save)"
    echo -e "   ✓ Local AI inference (DeepSeek R1 + Qwen Coder)"
    echo -e "   ✓ Smart routing (auto-detects code vs general queries)"
    echo -e "   ✓ Web search & crawling with PDF extraction"
    echo -e "   ✓ RAG knowledge base"
    echo -e "   ✓ Hardware attestation (DSMIL Mode 5)"
    echo -e "   ✓ 76.4 TOPS compute (NPU + GPU + NCS2)\n"

    echo -e "${CYAN}📖 Documentation:${NC}"
    echo -e "   README:    ${YELLOW}$INSTALL_DIR/README.md${NC}"
    echo -e "   Install:   ${YELLOW}$INSTALL_DIR/INSTALL.md${NC}"
    echo -e "   Structure: ${YELLOW}$INSTALL_DIR/STRUCTURE.md${NC}"
    echo -e "   Docs:      ${YELLOW}$INSTALL_DIR/00-documentation/${NC}\n"

    echo -e "${CYAN}🔐 Security Status:${NC}"
    if lsmod | grep -q dsmil; then
        echo -e "   DSMIL Framework: ${GREEN}✓ Active (Hardware Attestation Enabled)${NC}"
    else
        echo -e "   DSMIL Framework: ${YELLOW}⚠ Not Loaded (Software Mode)${NC}"
    fi
    echo -e "   TPM 2.0: $([ -c /dev/tpm0 ] && echo -e "${GREEN}✓ Available${NC}" || echo -e "${YELLOW}⚠ Not Found${NC}")"
    echo -e "   Mode: ${GREEN}STANDARD${NC} (Safe for training)\n"

    echo -e "${GREEN}Happy coding! 🎯${NC}\n"
}

#═══════════════════════════════════════════════════════════════════════════════
# Main Installation Flow
#═══════════════════════════════════════════════════════════════════════════════

main() {
    print_banner
    check_root

    echo -e "${YELLOW}This will install DSMIL Complete Platform v8.3.2${NC}"
    echo -e "${YELLOW}Components: Framework + AI System + Hardware Attestation${NC}"
    echo -e "${YELLOW}Installation directory: $INSTALL_DIR${NC}\n"

    echo -e "This installation includes:"
    echo -e "  • System dependencies (build tools, Python, etc.)"
    echo -e "  • DSMIL kernel framework (hardware attestation)"
    echo -e "  • Ollama AI runtime + models (~3-11GB download)"
    echo -e "  • Python AI/ML libraries"
    echo -e "  • Web interface + API server"
    echo -e "  • Systemd service (auto-start)\n"

    read -p "Continue with installation? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_warning "Installation cancelled"
        exit 0
    fi

    # Run installation steps
    check_system_requirements
    install_system_dependencies
    install_python_dependencies
    install_dsmil_framework
    install_ollama
    download_ai_models
    setup_hardware_optimization
    create_configuration
    setup_systemd_service

    # Verify installation
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
