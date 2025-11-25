#!/bin/bash
#
# TPM2 Early Boot Acceleration Module - Uninstallation Script
# Dell Latitude 5450 MIL-SPEC with Intel Core Ultra 7 165H
#
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
# Author: Claude Code TPM2 Acceleration Installer
# Version: 1.0.0

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Module configuration
MODULE_NAME="tpm2_accel_early"
KERNEL_VERSION=$(uname -r)

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script requires root privileges"
        echo "Please run: sudo $0"
        exit 1
    fi
}

unload_module() {
    print_header "Unloading Module"

    if lsmod | grep -q "^${MODULE_NAME}"; then
        print_info "Unloading ${MODULE_NAME}..."
        if modprobe -r ${MODULE_NAME} 2>&1; then
            print_success "Module unloaded"
        else
            print_warning "Module unload failed (may be in use)"
        fi
    else
        print_info "Module not currently loaded"
    fi
}

remove_module_files() {
    print_header "Removing Module Files"

    # Remove installed module
    MODULE_PATH="/lib/modules/${KERNEL_VERSION}/kernel/drivers/tpm/${MODULE_NAME}.ko"
    if [ -f "$MODULE_PATH" ]; then
        rm -f "$MODULE_PATH"
        print_success "Module removed: $MODULE_PATH"
    else
        print_info "Module file not found (already removed?)"
    fi

    # Update dependencies
    depmod -a
    print_success "Module dependencies updated"
}

remove_configurations() {
    print_header "Removing Configuration Files"

    # Remove modules-load.d
    if [ -f "/etc/modules-load.d/tpm2-acceleration.conf" ]; then
        rm -f /etc/modules-load.d/tpm2-acceleration.conf
        print_success "Removed /etc/modules-load.d/tpm2-acceleration.conf"
    fi

    # Remove modprobe.d
    if [ -f "/etc/modprobe.d/tpm2-acceleration.conf" ]; then
        rm -f /etc/modprobe.d/tpm2-acceleration.conf
        print_success "Removed /etc/modprobe.d/tpm2-acceleration.conf"
    fi

    # Disable and remove systemd service
    if systemctl is-enabled tpm2-acceleration-early.service &> /dev/null; then
        systemctl disable tpm2-acceleration-early.service &> /dev/null
        print_success "Systemd service disabled"
    fi

    if [ -f "/etc/systemd/system/tpm2-acceleration-early.service" ]; then
        rm -f /etc/systemd/system/tpm2-acceleration-early.service
        systemctl daemon-reload
        print_success "Systemd service removed"
    fi
}

update_initramfs() {
    print_header "Updating Initramfs"

    print_info "Updating initramfs to remove module..."
    if update-initramfs -u 2>&1 | grep -v "mkinitramfs: copy_file" | tail -5; then
        print_success "Initramfs updated"
    else
        print_warning "Initramfs update completed with warnings"
    fi
}

show_summary() {
    print_header "Uninstallation Complete"

    echo -e "${GREEN}✅ TPM2 Early Boot Acceleration Module Removed${NC}"
    echo ""
    echo "Removed Components:"
    echo "  • Kernel module (${MODULE_NAME}.ko)"
    echo "  • Early boot configuration"
    echo "  • Module parameters"
    echo "  • Systemd service"
    echo "  • Initramfs integration"
    echo ""
    echo "Verification:"
    echo "  • lsmod | grep tpm2_accel  (should be empty)"
    echo "  • ls /dev/tpm2_accel_early  (should not exist)"
    echo ""
    echo -e "${YELLOW}Note:${NC} A reboot may be required for complete cleanup"
    echo "      Device node may persist until reboot"
}

# Main execution
main() {
    print_header "TPM2 Early Boot Acceleration Module Uninstaller"
    echo ""

    check_root

    # Confirm uninstallation
    read -p "Are you sure you want to uninstall the TPM2 acceleration module? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Uninstallation cancelled"
        exit 0
    fi
    echo ""

    unload_module
    echo ""

    remove_module_files
    echo ""

    remove_configurations
    echo ""

    update_initramfs
    echo ""

    show_summary

    exit 0
}

# Run main function
main "$@"
