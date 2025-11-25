#!/bin/bash
# Thermal Guardian Installation Script
# Agent 3 Implementation for Dell LAT5150DRVMIL

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/thermal-guardian"
CONFIG_FILE="/etc/thermal_guardian.conf"
SERVICE_FILE="/etc/systemd/system/thermal-guardian.service"
LOG_FILE="/var/log/thermal_guardian.log"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_system_compatibility() {
    log_info "Checking system compatibility..."
    
    # Check for Dell system
    if ! dmidecode -s system-manufacturer 2>/dev/null | grep -qi dell; then
        log_warning "This system may not be a Dell computer"
        log_warning "Thermal Guardian will still work but Dell-specific features may not be available"
    fi
    
    # Check for required Python version
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 6) else 1)" 2>/dev/null; then
        log_error "Python 3.6 or higher is required"
        exit 1
    fi
    
    # Check for thermal zones
    if [[ ! -d /sys/class/thermal ]]; then
        log_error "No thermal zones found. System may not support thermal management."
        exit 1
    fi
    
    # Count available thermal sensors
    thermal_count=$(find /sys/class/thermal -name "thermal_zone*" -type d | wc -l)
    hwmon_count=$(find /sys/class/hwmon -name "hwmon*" -type d | wc -l)
    
    log_info "Found $thermal_count thermal zones and $hwmon_count hardware monitors"
    
    # Check for Intel P-State
    if [[ -d /sys/devices/system/cpu/intel_pstate ]]; then
        log_info "Intel P-State found - CPU frequency scaling available"
    else
        log_warning "Intel P-State not found - CPU frequency scaling may not work"
    fi
    
    # Check for Dell SMM
    if find /sys/class/hwmon -name "hwmon*" -exec cat {}/name \; 2>/dev/null | grep -q "dell_smm\|i8k"; then
        log_info "Dell SMM found - Dell fan control available"
    else
        log_warning "Dell SMM not found - Dell-specific fan control may not work"
    fi
}

install_dependencies() {
    log_info "Installing dependencies..."
    
    # Update package list
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y python3 python3-pip dmidecode lm-sensors
    elif command -v yum >/dev/null 2>&1; then
        yum install -y python3 python3-pip dmidecode lm_sensors
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y python3 python3-pip dmidecode lm_sensors
    else
        log_warning "Unknown package manager - please install python3, dmidecode, and lm-sensors manually"
    fi
    
    # Initialize sensors
    if command -v sensors-detect >/dev/null 2>&1; then
        log_info "Initializing hardware sensors..."
        yes | sensors-detect >/dev/null 2>&1 || true
    fi
}

create_directories() {
    log_info "Creating directories..."
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Set permissions
    chmod 755 "$INSTALL_DIR"
    touch "$LOG_FILE"
    chmod 644 "$LOG_FILE"
}

install_files() {
    log_info "Installing Thermal Guardian files..."
    
    # Copy main script
    cp "$SCRIPT_DIR/thermal_guardian.py" "$INSTALL_DIR/"
    chmod 755 "$INSTALL_DIR/thermal_guardian.py"
    
    # Copy configuration
    if [[ ! -f "$CONFIG_FILE" ]]; then
        cp "$SCRIPT_DIR/thermal_guardian.conf" "$CONFIG_FILE"
        chmod 644 "$CONFIG_FILE"
        log_success "Configuration file installed at $CONFIG_FILE"
    else
        log_warning "Configuration file already exists at $CONFIG_FILE - not overwriting"
        log_info "New configuration template available at $SCRIPT_DIR/thermal_guardian.conf"
    fi
    
    # Copy and install systemd service
    cp "$SCRIPT_DIR/thermal-guardian.service" "$SERVICE_FILE"
    
    # Update service file paths
    sed -i "s|/opt/github/LAT5150DRVMIL/thermal_guardian.py|$INSTALL_DIR/thermal_guardian.py|g" "$SERVICE_FILE"
    
    chmod 644 "$SERVICE_FILE"
}

configure_service() {
    log_info "Configuring systemd service..."
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable thermal-guardian.service
    
    log_success "Thermal Guardian service enabled"
}

test_installation() {
    log_info "Testing installation..."
    
    # Test basic functionality
    if ! python3 "$INSTALL_DIR/thermal_guardian.py" --test; then
        log_error "Installation test failed"
        return 1
    fi
    
    # Test service file
    if ! systemctl is-enabled thermal-guardian.service >/dev/null 2>&1; then
        log_error "Service is not enabled"
        return 1
    fi
    
    log_success "Installation test passed"
}

show_status() {
    log_info "Checking thermal sensors..."
    
    # Show available thermal sensors
    echo
    echo "Available thermal zones:"
    for zone in /sys/class/thermal/thermal_zone*; do
        if [[ -r "$zone/type" && -r "$zone/temp" ]]; then
            zone_type=$(cat "$zone/type")
            zone_temp=$(cat "$zone/temp")
            zone_temp_c=$((zone_temp / 1000))
            echo "  $(basename "$zone"): $zone_type (${zone_temp_c}°C)"
        fi
    done
    
    echo
    echo "Available hardware monitors:"
    for hwmon in /sys/class/hwmon/hwmon*; do
        if [[ -r "$hwmon/name" ]]; then
            hwmon_name=$(cat "$hwmon/name")
            echo "  $(basename "$hwmon"): $hwmon_name"
        fi
    done
    
    # Show current status
    echo
    log_info "Current thermal status:"
    python3 "$INSTALL_DIR/thermal_guardian.py" --status 2>/dev/null || log_warning "Could not get thermal status"
}

print_usage_info() {
    echo
    log_success "Thermal Guardian installation complete!"
    echo
    echo "Usage:"
    echo "  Start service:    sudo systemctl start thermal-guardian"
    echo "  Stop service:     sudo systemctl stop thermal-guardian"
    echo "  Service status:   sudo systemctl status thermal-guardian"
    echo "  View logs:        sudo journalctl -u thermal-guardian -f"
    echo "  Check status:     sudo python3 $INSTALL_DIR/thermal_guardian.py --status"
    echo
    echo "Configuration file: $CONFIG_FILE"
    echo "Log file:          $LOG_FILE"
    echo
    echo "To start monitoring immediately:"
    echo "  sudo systemctl start thermal-guardian"
}

# Main installation process
main() {
    log_info "Starting Thermal Guardian installation..."
    
    check_root
    check_system_compatibility
    install_dependencies
    create_directories
    install_files
    configure_service
    test_installation
    show_status
    print_usage_info
    
    log_success "Installation completed successfully!"
}

# Command line argument handling
case "${1:-install}" in
    "install")
        main
        ;;
    "uninstall")
        log_info "Uninstalling Thermal Guardian..."
        systemctl stop thermal-guardian.service 2>/dev/null || true
        systemctl disable thermal-guardian.service 2>/dev/null || true
        rm -f "$SERVICE_FILE"
        rm -rf "$INSTALL_DIR"
        rm -f "$CONFIG_FILE.bak"
        systemctl daemon-reload
        log_success "Thermal Guardian uninstalled"
        ;;
    "status")
        show_status
        ;;
    "test")
        test_installation
        ;;
    *)
        echo "Usage: $0 [install|uninstall|status|test]"
        exit 1
        ;;
esac