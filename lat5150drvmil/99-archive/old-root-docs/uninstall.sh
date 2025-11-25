#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# DSMIL Unified AI Platform - Uninstaller
# Version: 8.3
#═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

USER_HOME="$HOME"

print_banner() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  DSMIL UNIFIED AI PLATFORM - UNINSTALLER"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

main() {
    print_banner

    echo -e "${YELLOW}This will remove DSMIL Unified AI Platform${NC}"
    echo -e "${RED}WARNING: This action cannot be undone!${NC}\n"

    echo "The following will be removed:"
    echo "  - DSMIL systemd service"
    echo "  - Configuration files in ~/.config/dsmil"
    echo "  - RAG index data"
    echo ""
    echo "The following will be KEPT:"
    echo "  - Source code in $(pwd)"
    echo "  - Ollama and models (if you want to remove them, run: sudo apt remove ollama)"
    echo "  - Python packages"
    echo ""

    read -p "Continue with uninstall? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Uninstall cancelled"
        exit 0
    fi

    # Stop and disable service
    print_step "Stopping DSMIL service..."
    sudo systemctl stop dsmil-server.service 2>/dev/null || true
    sudo systemctl disable dsmil-server.service 2>/dev/null || true
    print_success "Service stopped"

    # Remove service file
    print_step "Removing systemd service..."
    sudo rm -f /etc/systemd/system/dsmil-server.service
    sudo systemctl daemon-reload
    print_success "Service file removed"

    # Remove config
    print_step "Removing configuration..."
    rm -rf "$USER_HOME/.config/dsmil"
    print_success "Configuration removed"

    # Remove RAG index
    print_step "Removing RAG index..."
    rm -rf "$USER_HOME/.local/share/dsmil"
    print_success "RAG index removed"

    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  UNINSTALL COMPLETE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}\n"

    echo -e "${CYAN}Source code remains in: $(pwd)${NC}"
    echo -e "${CYAN}To reinstall, run: ./install.sh${NC}\n"

    echo -e "${YELLOW}Optional cleanup:${NC}"
    echo -e "  Remove Ollama: ${BLUE}sudo apt remove ollama${NC}"
    echo -e "  Remove models: ${BLUE}rm -rf ~/.ollama${NC}"
    echo -e "  Remove Python packages: ${BLUE}pip3 uninstall requests anthropic flask${NC}\n"
}

main "$@"
