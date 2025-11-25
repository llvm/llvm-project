#!/bin/bash
#
# Installation script for Dell MIL-SPEC userspace utilities
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Default paths
PREFIX="${PREFIX:-/usr/local}"
BINDIR="${PREFIX}/bin"
SYSTEMD_DIR="/etc/systemd/system"
LOG_DIR="/var/log"

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}This script must be run as root${NC}"
        exit 1
    fi
}

# Check for required files
check_files() {
    local missing=0
    
    echo "Checking required files..."
    
    for file in milspec-monitor.c milspec-control.c dell-milspec.h Makefile.utils; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}Missing: $file${NC}"
            missing=1
        fi
    done
    
    if [ $missing -eq 1 ]; then
        echo -e "${RED}Please ensure all required files are present${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All required files found${NC}"
}

# Check for driver
check_driver() {
    echo "Checking for dell-milspec driver..."
    
    if ! lsmod | grep -q dell_milspec; then
        echo -e "${YELLOW}Warning: dell-milspec driver not loaded${NC}"
        echo "The utilities will not work until the driver is loaded"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}Driver is loaded${NC}"
    fi
}

# Build utilities
build_utils() {
    echo "Building utilities..."
    
    make -f Makefile.utils clean
    make -f Makefile.utils
    
    if [ ! -f milspec-monitor ] || [ ! -f milspec-control ]; then
        echo -e "${RED}Build failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Build successful${NC}"
}

# Install binaries
install_binaries() {
    echo "Installing binaries to $BINDIR..."
    
    mkdir -p "$BINDIR"
    install -m 755 milspec-monitor "$BINDIR/"
    install -m 755 milspec-control "$BINDIR/"
    
    echo -e "${GREEN}Binaries installed${NC}"
}

# Install systemd service
install_service() {
    echo "Installing systemd service..."
    
    if [ -f milspec-monitor.service ]; then
        install -m 644 milspec-monitor.service "$SYSTEMD_DIR/"
        
        # Update path in service file if needed
        if [ "$PREFIX" != "/usr/local" ]; then
            sed -i "s|/usr/local/bin|$BINDIR|g" "$SYSTEMD_DIR/milspec-monitor.service"
        fi
        
        systemctl daemon-reload
        echo -e "${GREEN}Systemd service installed${NC}"
        
        read -p "Enable milspec-monitor service to start at boot? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            systemctl enable milspec-monitor.service
            echo -e "${GREEN}Service enabled${NC}"
        fi
        
        read -p "Start milspec-monitor service now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            systemctl start milspec-monitor.service
            echo -e "${GREEN}Service started${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: milspec-monitor.service not found${NC}"
    fi
}

# Create log directory and file
setup_logging() {
    echo "Setting up logging..."
    
    touch "$LOG_DIR/milspec-events.log"
    chmod 640 "$LOG_DIR/milspec-events.log"
    
    echo -e "${GREEN}Log file created: $LOG_DIR/milspec-events.log${NC}"
}

# Test installation
test_installation() {
    echo "Testing installation..."
    
    if "$BINDIR/milspec-control" -h > /dev/null 2>&1; then
        echo -e "${GREEN}milspec-control: OK${NC}"
    else
        echo -e "${RED}milspec-control: FAILED${NC}"
    fi
    
    if "$BINDIR/milspec-monitor" -h > /dev/null 2>&1; then
        echo -e "${GREEN}milspec-monitor: OK${NC}"
    else
        echo -e "${RED}milspec-monitor: FAILED${NC}"
    fi
    
    # Try to get status if driver is loaded
    if lsmod | grep -q dell_milspec; then
        echo "Attempting to read MIL-SPEC status..."
        if "$BINDIR/milspec-control" status > /dev/null 2>&1; then
            echo -e "${GREEN}Driver communication: OK${NC}"
        else
            echo -e "${YELLOW}Driver communication: Failed (check permissions)${NC}"
        fi
    fi
}

# Uninstall function
uninstall() {
    echo "Uninstalling Dell MIL-SPEC utilities..."
    
    # Stop and disable service
    if systemctl is-active --quiet milspec-monitor.service; then
        systemctl stop milspec-monitor.service
    fi
    if systemctl is-enabled --quiet milspec-monitor.service; then
        systemctl disable milspec-monitor.service
    fi
    
    # Remove files
    rm -f "$BINDIR/milspec-monitor"
    rm -f "$BINDIR/milspec-control"
    rm -f "$SYSTEMD_DIR/milspec-monitor.service"
    
    systemctl daemon-reload
    
    echo -e "${GREEN}Uninstall complete${NC}"
}

# Main installation
main() {
    echo "Dell MIL-SPEC Utilities Installer"
    echo "================================="
    echo
    
    # Parse arguments
    if [ "$1" = "uninstall" ]; then
        check_root
        uninstall
        exit 0
    fi
    
    check_root
    check_files
    check_driver
    build_utils
    install_binaries
    install_service
    setup_logging
    test_installation
    
    echo
    echo -e "${GREEN}Installation complete!${NC}"
    echo
    echo "Usage:"
    echo "  milspec-control status    - Show MIL-SPEC status"
    echo "  milspec-monitor -tc       - Monitor events with color"
    echo
    echo "To uninstall:"
    echo "  $0 uninstall"
    echo
}

main "$@"
