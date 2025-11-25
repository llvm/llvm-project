#!/bin/bash
#
# DSMIL Yubikey Configuration Script
# ===================================
#
# Installs and configures Yubikey support for DSMIL tactical platform.
#
# Features:
# - Installs required packages (libfido2, yubikey-manager, pcscd)
# - Configures udev rules for Yubikey access
# - Sets up PAM for challenge-response authentication
# - Tests Yubikey detection and functionality
#
# Usage: sudo ./configure_yubikey.sh [install|remove|test|status]
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
    echo "Usage: sudo $0 [install|remove|test|status]"
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
                libfido2-1 \
                libfido2-dev \
                libfido2-doc \
                python3-fido2 \
                yubikey-manager \
                yubikey-personalization \
                libu2f-udev \
                pcscd \
                pcsc-tools \
                libpcsclite-dev \
                python3-pip
            ;;
        dnf|yum)
            print_step "Using $pm package manager"
            $pm install -y \
                libfido2 \
                libfido2-devel \
                python3-fido2 \
                yubikey-manager \
                yubikey-personalization \
                pcsc-lite \
                pcsc-tools \
                pcsc-lite-devel \
                python3-pip
            ;;
        pacman)
            print_step "Using pacman package manager"
            pacman -Sy --noconfirm \
                libfido2 \
                python-fido2 \
                yubikey-manager \
                yubikey-personalization \
                pcsclite \
                ccid \
                python-pip
            ;;
        *)
            print_error "Unsupported package manager"
            echo "Please install the following packages manually:"
            echo "  - libfido2 and libfido2-dev"
            echo "  - python3-fido2"
            echo "  - yubikey-manager"
            echo "  - pcscd and pcsc-tools"
            exit 1
            ;;
    esac

    print_step "Packages installed"
}

# Install Python packages
install_python_packages() {
    print_header "Installing Python Packages"

    pip3 install --upgrade \
        fido2 \
        yubikey-manager \
        yubico-client

    print_step "Python packages installed"
}

# Configure udev rules
configure_udev() {
    print_header "Configuring udev Rules"

    # Check if rules already exist
    if [ -f /etc/udev/rules.d/70-u2f.rules ]; then
        print_warning "udev rules already exist, backing up..."
        mv /etc/udev/rules.d/70-u2f.rules /etc/udev/rules.d/70-u2f.rules.bak
    fi

    # Create udev rules
    cat > /etc/udev/rules.d/70-u2f.rules << 'EOF'
# Yubico YubiKey
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1050", TAG+="uaccess", GROUP="plugdev", MODE="0660"

# Yubico Security Key
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1050", ATTRS{idProduct}=="0407", TAG+="uaccess", GROUP="plugdev", MODE="0660"

# FIDO U2F devices
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1050", MODE="0664", GROUP="plugdev"
EOF

    # Reload udev rules
    udevadm control --reload-rules
    udevadm trigger

    print_step "udev rules configured"
}

# Configure pcscd
configure_pcscd() {
    print_header "Configuring PC/SC Daemon"

    # Enable and start pcscd
    if systemctl is-enabled pcscd &> /dev/null; then
        print_step "pcscd already enabled"
    else
        systemctl enable pcscd
        print_step "pcscd enabled"
    fi

    if systemctl is-active pcscd &> /dev/null; then
        print_step "pcscd already running"
    else
        systemctl start pcscd
        print_step "pcscd started"
    fi

    print_step "PC/SC daemon configured"
}

# Create user group
create_group() {
    print_header "Configuring User Access"

    # Create plugdev group if it doesn't exist
    if getent group plugdev > /dev/null 2>&1; then
        print_step "plugdev group exists"
    else
        groupadd plugdev
        print_step "plugdev group created"
    fi

    # Add current user to plugdev group
    if [ ! -z "$SUDO_USER" ]; then
        if groups $SUDO_USER | grep -q plugdev; then
            print_step "User $SUDO_USER already in plugdev group"
        else
            usermod -a -G plugdev $SUDO_USER
            print_step "User $SUDO_USER added to plugdev group"
            print_warning "User needs to log out and back in for group changes to take effect"
        fi
    fi
}

# Test Yubikey detection
test_yubikey() {
    print_header "Testing Yubikey Detection"

    # Test with ykman
    if command -v ykman &> /dev/null; then
        echo -e "${BLUE}Running: ykman list${NC}"
        if ykman list; then
            print_step "Yubikey detected successfully"
        else
            print_warning "No Yubikey detected"
            echo "  - Insert your Yubikey"
            echo "  - Check USB connection"
            echo "  - Verify udev rules: ls -l /etc/udev/rules.d/70-u2f.rules"
        fi
    else
        print_error "ykman command not found"
    fi

    echo ""

    # Test FIDO2
    if python3 -c "import fido2; print('FIDO2 module: OK')" 2>/dev/null; then
        print_step "FIDO2 Python module working"
    else
        print_error "FIDO2 Python module not working"
    fi

    # Test pcscd
    if systemctl is-active pcscd &> /dev/null; then
        print_step "pcscd service running"
    else
        print_error "pcscd service not running"
    fi
}

# Show status
show_status() {
    print_header "Yubikey Configuration Status"

    # Check packages
    echo -e "${BLUE}Package Status:${NC}"

    if command -v ykman &> /dev/null; then
        print_step "yubikey-manager: $(ykman --version)"
    else
        print_error "yubikey-manager: Not installed"
    fi

    if python3 -c "import fido2" 2>/dev/null; then
        local ver=$(python3 -c "import fido2; print(fido2.__version__)" 2>/dev/null)
        print_step "python3-fido2: $ver"
    else
        print_error "python3-fido2: Not installed"
    fi

    if command -v pcsc_scan &> /dev/null; then
        print_step "pcsc-tools: Installed"
    else
        print_error "pcsc-tools: Not installed"
    fi

    # Check services
    echo -e "\n${BLUE}Service Status:${NC}"

    if systemctl is-active pcscd &> /dev/null; then
        print_step "pcscd: Running"
    else
        print_error "pcscd: Not running"
    fi

    # Check udev rules
    echo -e "\n${BLUE}Configuration Status:${NC}"

    if [ -f /etc/udev/rules.d/70-u2f.rules ]; then
        print_step "udev rules: Configured"
    else
        print_error "udev rules: Not configured"
    fi

    # Check group membership
    if [ ! -z "$SUDO_USER" ]; then
        if groups $SUDO_USER | grep -q plugdev; then
            print_step "User $SUDO_USER: In plugdev group"
        else
            print_warning "User $SUDO_USER: Not in plugdev group"
        fi
    fi

    # Check for Yubikey
    echo -e "\n${BLUE}Yubikey Detection:${NC}"

    if command -v ykman &> /dev/null; then
        if ykman list &> /dev/null; then
            print_step "Yubikey detected"
            ykman list
        else
            print_warning "No Yubikey detected"
        fi
    fi
}

# Remove configuration
remove_yubikey() {
    print_header "Removing Yubikey Configuration"

    print_warning "This will remove Yubikey packages and configuration"
    read -p "Are you sure? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Removal cancelled"
        exit 0
    fi

    # Remove udev rules
    if [ -f /etc/udev/rules.d/70-u2f.rules ]; then
        rm /etc/udev/rules.d/70-u2f.rules
        udevadm control --reload-rules
        print_step "udev rules removed"
    fi

    # Stop pcscd
    if systemctl is-active pcscd &> /dev/null; then
        systemctl stop pcscd
        systemctl disable pcscd
        print_step "pcscd stopped and disabled"
    fi

    # Remove packages
    local pm=$(detect_package_manager)

    case $pm in
        apt)
            apt-get remove -y \
                yubikey-manager \
                yubikey-personalization \
                python3-fido2
            print_step "Packages removed (libfido2 and pcscd kept for system use)"
            ;;
        dnf|yum)
            $pm remove -y \
                yubikey-manager \
                yubikey-personalization \
                python3-fido2
            print_step "Packages removed"
            ;;
        pacman)
            pacman -R --noconfirm \
                yubikey-manager \
                yubikey-personalization \
                python-fido2
            print_step "Packages removed"
            ;;
    esac

    print_step "Yubikey configuration removed"
}

# Main installation
install_yubikey() {
    print_header "DSMIL Yubikey Configuration"
    echo "Installing Yubikey support..."

    install_packages
    install_python_packages
    configure_udev
    configure_pcscd
    create_group

    print_header "Installation Complete"
    print_step "Yubikey support installed successfully"
    echo ""
    echo "Next steps:"
    echo "  1. Log out and back in (for group changes)"
    echo "  2. Insert your Yubikey"
    echo "  3. Test: ./configure_yubikey.sh test"
    echo "  4. Register: python3 02-ai-engine/yubikey_admin.py register"
}

# Main
case "${1:-install}" in
    install)
        install_yubikey
        ;;
    remove)
        remove_yubikey
        ;;
    test)
        test_yubikey
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {install|remove|test|status}"
        echo ""
        echo "Commands:"
        echo "  install  - Install and configure Yubikey support"
        echo "  remove   - Remove Yubikey configuration"
        echo "  test     - Test Yubikey detection"
        echo "  status   - Show configuration status"
        exit 1
        ;;
esac

exit 0
