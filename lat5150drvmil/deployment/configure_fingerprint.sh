#!/bin/bash
#
# DSMIL Fingerprint Reader Configuration Script
# ==============================================
#
# Installs and configures fingerprint reader support for DSMIL tactical platform.
#
# Features:
# - Installs required packages (fprintd, libfprint, PAM modules)
# - Configures PAM for fingerprint authentication
# - Sets up user permissions
# - Tests fingerprint reader detection and functionality
#
# Usage: sudo ./configure_fingerprint.sh [install|remove|test|status|pam-enable|pam-disable]
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ] && [ "$1" != "status" ] && [ "$1" != "test" ]; then
    echo -e "${RED}Error: This script must be run as root${NC}"
    echo "Usage: sudo $0 [install|remove|test|status|pam-enable|pam-disable]"
    exit 1
fi

# Helper functions
print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Detect package manager
detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        echo "apt"
    elif command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v yum &> /dev/null; then
        echo "yum"
    elif command -v pacman &> /dev/null; then
        echo "pacman"
    else
        echo "unknown"
    fi
}

# Install packages
install_packages() {
    local pm=$(detect_package_manager)

    print_header "Installing Required Packages"

    case $pm in
        apt)
            print_step "Using apt package manager"
            apt-get update
            apt-get install -y \
                fprintd \
                libpam-fprintd \
                libfprint-2-2 \
                python3-dbus \
                python3-gi \
                gir1.2-glib-2.0
            ;;
        dnf|yum)
            print_step "Using $pm package manager"
            $pm install -y \
                fprintd \
                fprintd-pam \
                libfprint \
                python3-dbus \
                python3-gobject
            ;;
        pacman)
            print_step "Using pacman package manager"
            pacman -Sy --noconfirm \
                fprintd \
                libfprint \
                python-dbus \
                python-gobject
            ;;
        *)
            print_error "Unsupported package manager"
            echo "Please install the following packages manually:"
            echo "  - fprintd"
            echo "  - libpam-fprintd (or fprintd-pam)"
            echo "  - libfprint"
            echo "  - python3-dbus"
            exit 1
            ;;
    esac

    print_step "Packages installed"
}

# Configure fprintd service
configure_fprintd() {
    print_header "Configuring fprintd Service"

    # Enable and start fprintd
    if systemctl is-enabled fprintd &> /dev/null; then
        print_step "fprintd already enabled"
    else
        systemctl enable fprintd
        print_step "fprintd enabled"
    fi

    if systemctl is-active fprintd &> /dev/null; then
        print_step "fprintd already running"
    else
        systemctl start fprintd
        print_step "fprintd started"
    fi

    # Check D-Bus service
    if dbus-send --system --dest=net.reactivated.Fprint --type=method_call --print-reply /net/reactivated/Fprint/Manager net.reactivated.Fprint.Manager.GetDevices &> /dev/null; then
        print_step "fprintd D-Bus service responding"
    else
        print_warning "fprintd D-Bus service not responding (may need reboot)"
    fi

    print_step "fprintd service configured"
}

# Configure PAM
configure_pam() {
    print_header "Configuring PAM for Fingerprint Authentication"

    print_warning "PAM configuration can affect system login!"
    print_warning "This will add fingerprint as an authentication option."
    print_warning "Password authentication will still be available as fallback."

    read -p "Configure PAM? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "PAM configuration skipped"
        return 0
    fi

    # Backup PAM configs
    backup_dir="/etc/pam.d/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    cp /etc/pam.d/common-auth "$backup_dir/" 2>/dev/null || true
    cp /etc/pam.d/sudo "$backup_dir/" 2>/dev/null || true
    print_step "PAM configs backed up to $backup_dir"

    # Add fingerprint to common-auth (Debian/Ubuntu)
    if [ -f /etc/pam.d/common-auth ]; then
        if ! grep -q "pam_fprintd.so" /etc/pam.d/common-auth; then
            # Add after pam_unix.so line
            sed -i '/pam_unix.so/a auth    [success=1 default=ignore]    pam_fprintd.so max_tries=3 timeout=10' /etc/pam.d/common-auth
            print_step "Added fingerprint to /etc/pam.d/common-auth"
        else
            print_step "Fingerprint already configured in common-auth"
        fi
    fi

    # Add fingerprint to sudo
    if [ -f /etc/pam.d/sudo ]; then
        if ! grep -q "pam_fprintd.so" /etc/pam.d/sudo; then
            # Add at the top, before @include common-auth
            sed -i '1i auth    sufficient    pam_fprintd.so max_tries=3 timeout=10' /etc/pam.d/sudo
            print_step "Added fingerprint to /etc/pam.d/sudo"
        else
            print_step "Fingerprint already configured in sudo"
        fi
    fi

    print_step "PAM configuration complete"
    print_warning "You may need to log out and back in for changes to take effect"
}

# Disable PAM configuration
disable_pam() {
    print_header "Disabling PAM Fingerprint Authentication"

    print_warning "This will remove fingerprint authentication from PAM."
    read -p "Continue? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled"
        return 0
    fi

    # Remove from common-auth
    if [ -f /etc/pam.d/common-auth ]; then
        if grep -q "pam_fprintd.so" /etc/pam.d/common-auth; then
            sed -i '/pam_fprintd.so/d' /etc/pam.d/common-auth
            print_step "Removed fingerprint from /etc/pam.d/common-auth"
        fi
    fi

    # Remove from sudo
    if [ -f /etc/pam.d/sudo ]; then
        if grep -q "pam_fprintd.so" /etc/pam.d/sudo; then
            sed -i '/pam_fprintd.so/d' /etc/pam.d/sudo
            print_step "Removed fingerprint from /etc/pam.d/sudo"
        fi
    fi

    print_step "PAM fingerprint authentication disabled"
}

# Test fingerprint reader
test_fingerprint() {
    print_header "Testing Fingerprint Reader"

    # Check if fprintd is running
    if ! systemctl is-active fprintd &> /dev/null; then
        print_error "fprintd service is not running"
        echo "Start it with: sudo systemctl start fprintd"
        return 1
    fi

    print_step "fprintd service is running"

    # List devices using fprintd
    echo -e "\n${BLUE}Running: fprintd-list (checking for devices)${NC}"

    if command -v fprintd-list &> /dev/null; then
        if fprintd-list 2>&1 | grep -q "No devices"; then
            print_warning "No fingerprint devices found"
            echo ""
            echo "Troubleshooting:"
            echo "  1. Check USB devices: lsusb | grep -i fingerprint"
            echo "  2. Check if device is recognized by kernel: dmesg | grep -i fprint"
            echo "  3. Check supported devices: /usr/share/doc/libfprint*/supported-devices.txt"
            echo "  4. Restart fprintd: sudo systemctl restart fprintd"
        else
            fprintd-list
            print_step "Fingerprint device(s) detected"
        fi
    else
        print_error "fprintd-list command not found"
    fi

    echo ""

    # Check Python D-Bus
    if python3 -c "import dbus; print('D-Bus module: OK')" 2>/dev/null; then
        print_step "Python D-Bus module working"
    else
        print_error "Python D-Bus module not working"
    fi

    # Test D-Bus connection
    echo -e "\n${BLUE}Testing D-Bus connection to fprintd${NC}"
    if python3 << 'EOF'
import dbus
try:
    bus = dbus.SystemBus()
    manager = bus.get_object('net.reactivated.Fprint', '/net/reactivated/Fprint/Manager')
    devices = manager.GetDevices(dbus_interface='net.reactivated.Fprint.Manager')
    if devices:
        print(f"Found {len(devices)} device(s) via D-Bus")
        for dev in devices:
            print(f"  - {dev}")
    else:
        print("No devices found via D-Bus")
except Exception as e:
    print(f"D-Bus error: {e}")
    exit(1)
EOF
    then
        print_step "D-Bus connection successful"
    else
        print_error "D-Bus connection failed"
    fi
}

# Show status
show_status() {
    print_header "Fingerprint Reader Configuration Status"

    # Check packages
    echo -e "${BLUE}Package Status:${NC}"

    if command -v fprintd-list &> /dev/null; then
        print_step "fprintd: Installed"
    else
        print_error "fprintd: Not installed"
    fi

    if ldconfig -p | grep -q libfprint; then
        print_step "libfprint: Installed"
    else
        print_error "libfprint: Not installed"
    fi

    if [ -f /lib/security/pam_fprintd.so ] || [ -f /usr/lib/security/pam_fprintd.so ]; then
        print_step "pam_fprintd: Installed"
    else
        print_error "pam_fprintd: Not installed"
    fi

    if python3 -c "import dbus" 2>/dev/null; then
        print_step "python3-dbus: Installed"
    else
        print_error "python3-dbus: Not installed"
    fi

    # Check services
    echo -e "\n${BLUE}Service Status:${NC}"

    if systemctl is-active fprintd &> /dev/null; then
        print_step "fprintd: Running"
    else
        print_error "fprintd: Not running"
    fi

    if systemctl is-enabled fprintd &> /dev/null; then
        print_step "fprintd: Enabled (starts at boot)"
    else
        print_warning "fprintd: Not enabled"
    fi

    # Check PAM configuration
    echo -e "\n${BLUE}PAM Configuration:${NC}"

    if [ -f /etc/pam.d/common-auth ]; then
        if grep -q "pam_fprintd.so" /etc/pam.d/common-auth; then
            print_step "Fingerprint enabled in common-auth"
        else
            print_warning "Fingerprint not configured in common-auth"
        fi
    fi

    if [ -f /etc/pam.d/sudo ]; then
        if grep -q "pam_fprintd.so" /etc/pam.d/sudo; then
            print_step "Fingerprint enabled for sudo"
        else
            print_warning "Fingerprint not configured for sudo"
        fi
    fi

    # Check for devices
    echo -e "\n${BLUE}Device Detection:${NC}"

    if command -v fprintd-list &> /dev/null; then
        if fprintd-list 2>&1 | grep -q "No devices"; then
            print_warning "No fingerprint devices detected"
        else
            print_step "Fingerprint device(s) detected"
            echo ""
            fprintd-list
        fi
    fi

    # Check enrolled prints
    echo -e "\n${BLUE}Enrolled Fingerprints:${NC}"

    if [ ! -z "$SUDO_USER" ]; then
        enrolled=$(fprintd-list "$SUDO_USER" 2>/dev/null | grep -c "finger" || echo "0")
        if [ "$enrolled" -gt 0 ]; then
            print_step "User $SUDO_USER has $enrolled fingerprint(s) enrolled"
        else
            print_warning "User $SUDO_USER has no fingerprints enrolled"
        fi
    fi
}

# Remove configuration
remove_fingerprint() {
    print_header "Removing Fingerprint Configuration"

    print_warning "This will remove fingerprint packages and configuration"
    read -p "Are you sure? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Removal cancelled"
        exit 0
    fi

    # Remove PAM configuration first
    disable_pam

    # Stop fprintd
    if systemctl is-active fprintd &> /dev/null; then
        systemctl stop fprintd
        systemctl disable fprintd
        print_step "fprintd stopped and disabled"
    fi

    # Remove packages
    local pm=$(detect_package_manager)

    case $pm in
        apt)
            apt-get remove -y fprintd libpam-fprintd
            print_step "Packages removed"
            ;;
        dnf|yum)
            $pm remove -y fprintd fprintd-pam
            print_step "Packages removed"
            ;;
        pacman)
            pacman -R --noconfirm fprintd
            print_step "Packages removed"
            ;;
    esac

    print_step "Fingerprint configuration removed"
}

# Main installation
install_fingerprint() {
    print_header "DSMIL Fingerprint Reader Configuration"
    echo "Installing fingerprint reader support..."

    install_packages
    configure_fprintd

    print_header "Installation Complete"
    print_step "Fingerprint reader support installed successfully"
    echo ""
    echo "Next steps:"
    echo "  1. Test device detection: ./configure_fingerprint.sh test"
    echo "  2. Enroll fingerprints: fprintd-enroll <username>"
    echo "  3. Or use admin tool: python3 02-ai-engine/fingerprint_admin.py enroll --user <username>"
    echo "  4. (Optional) Enable PAM: ./configure_fingerprint.sh pam-enable"
    echo ""
    echo "PAM integration allows fingerprint authentication for:"
    echo "  - sudo commands"
    echo "  - System login (gdm, lightdm, etc.)"
    echo "  - SSH (with proper configuration)"
}

# Main
case "${1:-install}" in
    install)
        install_fingerprint
        ;;
    remove)
        remove_fingerprint
        ;;
    test)
        test_fingerprint
        ;;
    status)
        show_status
        ;;
    pam-enable)
        configure_pam
        ;;
    pam-disable)
        disable_pam
        ;;
    *)
        echo "Usage: $0 {install|remove|test|status|pam-enable|pam-disable}"
        echo ""
        echo "Commands:"
        echo "  install      - Install and configure fingerprint reader support"
        echo "  remove       - Remove fingerprint configuration"
        echo "  test         - Test fingerprint reader detection"
        echo "  status       - Show configuration status"
        echo "  pam-enable   - Enable PAM integration (system login/sudo)"
        echo "  pam-disable  - Disable PAM integration"
        exit 1
        ;;
esac

exit 0
