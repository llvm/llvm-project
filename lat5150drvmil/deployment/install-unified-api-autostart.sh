#!/bin/bash
#
# Install LAT5150 Unified Tactical API with Atomic Red Team Auto-Start Service
# Configures SystemD to automatically start unified API on boot with full integration
#
# Features:
# - Unified Natural Language API on port 80
# - Atomic Red Team MITRE ATT&CK integration
# - 20 system capabilities (4 atomic + 16 existing)
# - Auto-start on boot
# - Auto-restart on failure
#
# Usage:
#   sudo ./install-unified-api-autostart.sh install
#   sudo ./install-unified-api-autostart.sh remove
#   sudo ./install-unified-api-autostart.sh status
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AI_ENGINE_DIR="${PROJECT_ROOT}/02-ai-engine"
WEB_INTERFACE_DIR="${PROJECT_ROOT}/03-web-interface"
MCP_DATA_DIR="${PROJECT_ROOT}/03-mcp-servers/atomic-red-team-data"
SERVICE_FILE="${SCRIPT_DIR}/systemd/lat5150-unified-api.service"
SERVICE_NAME="lat5150-unified-api.service"
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

    # Check unified API exists
    if [ ! -f "${WEB_INTERFACE_DIR}/unified_tactical_api.py" ]; then
        log_error "Unified Tactical API not found"
        exit 1
    fi

    # Check Atomic Red Team API exists
    if [ ! -f "${AI_ENGINE_DIR}/atomic_red_team_api.py" ]; then
        log_error "Atomic Red Team API not found"
        exit 1
    fi

    # Check Python dependencies
    log_info "Checking Python packages..."

    if ! python3 -c "import flask" 2>/dev/null; then
        log_warn "Flask not installed, installing..."
        pip3 install flask flask-cors
    fi

    if ! python3 -c "import flask_cors" 2>/dev/null; then
        log_warn "Flask-CORS not installed, installing..."
        pip3 install flask-cors
    fi

    # Check uv/uvx for Atomic Red Team MCP server
    if ! command -v uvx &> /dev/null && ! command -v uv &> /dev/null; then
        log_warn "uvx not found, installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"

        # Verify installation
        if ! command -v uvx &> /dev/null; then
            log_error "Failed to install uvx"
            exit 1
        fi
    fi

    # Check Atomic Red Team MCP server
    log_info "Checking Atomic Red Team MCP server..."
    if ! uvx --help &> /dev/null; then
        log_error "uvx is not working correctly"
        exit 1
    fi

    # Create data directory for Atomic Red Team
    log_info "Creating Atomic Red Team data directory..."
    mkdir -p "${MCP_DATA_DIR}"
    chown -R $(logname):$(logname) "${MCP_DATA_DIR}" 2>/dev/null || true

    log_info "✓ Dependencies OK"
}

install_service() {
    check_root
    check_dependencies

    log_section "Installing LAT5150 Unified API with Atomic Red Team Auto-Start"
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
    sleep 3

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
    log_info "🚀 Unified Tactical API Features:"
    log_info "   • Natural Language Interface (port 80)"
    log_info "   • Atomic Red Team (MITRE ATT&CK)"
    log_info "   • 20 System Capabilities"
    log_info "   • Auto-start on boot"
    log_info "   • Auto-restart on failure"
    echo ""
    log_info "Access at: http://localhost:80"
    log_info "Documentation: ${WEB_INTERFACE_DIR}/ATOMIC_RED_TEAM_NL_USAGE.md"
    echo ""
    log_info "Example queries:"
    log_info '  curl -X POST http://localhost/api/query -H "Content-Type: application/json" -d '"'"'{"query": "Show atomic tests for T1059.002"}'"'"
    log_info '  curl -X POST http://localhost/api/query -H "Content-Type: application/json" -d '"'"'{"query": "List all MITRE ATT&CK techniques"}'"'"
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

    log_section "Removing LAT5150 Unified API Auto-Start Service"
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
    log_section "LAT5150 Unified API Service Status"
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
    journalctl -u "$SERVICE_NAME" -n 15 --no-pager

    echo ""
    log_info "Enabled: $(systemctl is-enabled $SERVICE_NAME 2>/dev/null || echo 'disabled')"
    log_info "Active:  $(systemctl is-active $SERVICE_NAME 2>/dev/null || echo 'inactive')"
    echo ""

    # Test API if running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Testing API endpoint..."
        if curl -s http://localhost:80/api/self-awareness > /dev/null 2>&1; then
            log_info "✓ API is responding"

            # Check if Atomic Red Team is loaded
            if curl -s http://localhost:80/api/self-awareness | grep -q '"atomic_red_team": true'; then
                log_info "✓ Atomic Red Team: Loaded"
            else
                log_warn "✗ Atomic Red Team: Not loaded"
            fi
        else
            log_warn "✗ API not responding"
        fi
    fi
    echo ""
}

show_help() {
    cat <<EOF
Install LAT5150 Unified Tactical API with Atomic Red Team Auto-Start

Usage:
    sudo $0 <command>

Commands:
    install     Install and enable auto-start service
    remove      Remove auto-start service
    status      Show service status and test API
    help        Show this help message

Examples:
    # Install auto-start
    sudo $0 install

    # Check status
    sudo $0 status

    # Remove auto-start
    sudo $0 remove

Description:
    Configures SystemD to automatically start the LAT5150 Unified
    Tactical API on system boot. The service includes:

    - Natural Language Interface (port 80)
    - Atomic Red Team MITRE ATT&CK integration
    - 20 System Capabilities
    - Automatic start on boot
    - Automatic restart on failure
    - Resource limits and security hardening
    - Journal logging

Configuration:
    Service file: $SERVICE_PATH
    Working dir:  ${WEB_INTERFACE_DIR}
    Port:         80
    Features:     NL processing, Atomic Red Team, MCP servers
    Capabilities: 20 total (4 atomic + 16 system)

Dependencies:
    - Python 3.10+
    - Flask, Flask-CORS
    - uv/uvx package manager
    - atomic-red-team-mcp server

Post-Installation:
    - Service starts automatically on boot
    - Access interface at: http://localhost:80
    - Monitor logs: sudo journalctl -u $SERVICE_NAME -f
    - Test queries: See ATOMIC_RED_TEAM_NL_USAGE.md

Natural Language Query Examples:
    curl -X POST http://localhost/api/query \\
      -H "Content-Type: application/json" \\
      -d '{"query": "Show me atomic tests for T1059.002"}'

    curl -X POST http://localhost/api/query \\
      -H "Content-Type: application/json" \\
      -d '{"query": "Find mshta atomics for Windows"}'

    curl -X POST http://localhost/api/query \\
      -H "Content-Type: application/json" \\
      -d '{"query": "List all MITRE ATT&CK techniques"}'

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
