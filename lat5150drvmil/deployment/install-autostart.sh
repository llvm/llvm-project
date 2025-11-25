#!/bin/bash
#
# Install LAT5150 Tactical Interface Auto-Start Service
# Configures SystemD to automatically start tactical interface on boot
#
# Usage:
#   sudo ./install-autostart.sh install
#   sudo ./install-autostart.sh remove
#   sudo ./install-autostart.sh status
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="${SCRIPT_DIR}/systemd/lat5150-tactical.service"
SERVICE_NAME="lat5150-tactical.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "${CYAN}[====]${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Python3
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi

    # Check tactical interface exists
    if [ ! -f "/home/user/LAT5150DRVMIL/03-web-interface/secured_self_coding_api.py" ]; then
        log_error "Tactical interface not found"
        exit 1
    fi

    # Check Python dependencies
    if ! python3 -c "import flask" 2>/dev/null; then
        log_warn "Flask not installed, installing..."
        pip3 install flask flask-sock
    fi

    log_info "✓ Dependencies OK"
}

install_service() {
    check_root
    check_dependencies

    log_section "Installing LAT5150 Tactical Auto-Start Service"
    echo ""

    # Copy service file
    log_info "Installing SystemD service..."
    cp "$SERVICE_FILE" "$SERVICE_PATH"
    chmod 644 "$SERVICE_PATH"

    # Reload SystemD
    log_info "Reloading SystemD daemon..."
    systemctl daemon-reload

    # Enable service
    log_info "Enabling service..."
    systemctl enable "$SERVICE_NAME"

    # Start service
    log_info "Starting service..."
    systemctl start "$SERVICE_NAME"

    # Wait a moment for service to start
    sleep 2

    # Check status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "✓ Service started successfully"
    else
        log_error "Service failed to start"
        log_warn "Check logs with: journalctl -u $SERVICE_NAME -n 50"
        exit 1
    fi

    echo ""
    log_section "Installation Complete"
    echo ""
    log_info "Service: $SERVICE_NAME"
    log_info "Status:  $(systemctl is-active $SERVICE_NAME)"
    log_info "Enabled: $(systemctl is-enabled $SERVICE_NAME)"
    echo ""
    log_info "The tactical interface will now start automatically on boot"
    log_info "Access at: http://localhost:5001"
    echo ""
    log_info "Useful commands:"
    log_info "  sudo systemctl status $SERVICE_NAME"
    log_info "  sudo systemctl stop $SERVICE_NAME"
    log_info "  sudo systemctl restart $SERVICE_NAME"
    log_info "  sudo journalctl -u $SERVICE_NAME -f"
    echo ""
}

remove_service() {
    check_root

    log_section "Removing LAT5150 Tactical Auto-Start Service"
    echo ""

    # Stop service
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Stopping service..."
        systemctl stop "$SERVICE_NAME"
    fi

    # Disable service
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        log_info "Disabling service..."
        systemctl disable "$SERVICE_NAME"
    fi

    # Remove service file
    if [ -f "$SERVICE_PATH" ]; then
        log_info "Removing service file..."
        rm "$SERVICE_PATH"
    fi

    # Reload SystemD
    log_info "Reloading SystemD daemon..."
    systemctl daemon-reload
    systemctl reset-failed 2>/dev/null || true

    echo ""
    log_info "✓ Service removed"
    echo ""
}

show_status() {
    log_section "LAT5150 Tactical Service Status"
    echo ""

    if [ -f "$SERVICE_PATH" ]; then
        log_info "Service file: Installed"
    else
        log_warn "Service file: Not installed"
        return
    fi

    # Service status
    echo "Status:"
    systemctl status "$SERVICE_NAME" --no-pager | head -20

    echo ""
    echo "Recent logs:"
    journalctl -u "$SERVICE_NAME" -n 10 --no-pager

    echo ""
    log_info "Enabled: $(systemctl is-enabled $SERVICE_NAME 2>/dev/null || echo 'disabled')"
    log_info "Active:  $(systemctl is-active $SERVICE_NAME 2>/dev/null || echo 'inactive')"
    echo ""
}

show_help() {
    cat <<EOF
Install LAT5150 Tactical Interface Auto-Start Service

Usage:
    sudo $0 <command>

Commands:
    install     Install and enable auto-start service
    remove      Remove auto-start service
    status      Show service status
    help        Show this help message

Examples:
    # Install auto-start
    sudo $0 install

    # Check status
    sudo $0 status

    # Remove auto-start
    sudo $0 remove

Description:
    Configures SystemD to automatically start the LAT5150 Tactical
    Interface on system boot. The service includes:

    - Automatic start on boot
    - Automatic restart on failure
    - Resource limits and security hardening
    - Journal logging

Configuration:
    Service file: $SERVICE_PATH
    Working dir:  /home/user/LAT5150DRVMIL
    Port:         5001
    Security:     HIGH level
    Features:     RAG, INT8, Learning enabled

Post-Installation:
    - Service starts automatically on boot
    - Access interface at: http://localhost:5001
    - Monitor logs: sudo journalctl -u $SERVICE_NAME -f

EOF
}

# Main
case "${1:-help}" in
    install)
        install_service
        ;;
    remove)
        remove_service
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
