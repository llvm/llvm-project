#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# ZFSBootMenu Recovery - Fix Missing Boot Environment
# Restores livecd-xen-ai boot environment to ZFSBootMenu
#═══════════════════════════════════════════════════════════════════════════════

set -e

# Configuration
SUDO_PASS="1786"
ZFS_PASS="1/0523/600260"
POOL="rpool"
BE_NAME="livecd-xen-ai"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_banner() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ZFSBootMenu Recovery - Fix Missing Boot Environment"
    echo "═══════════════════════════════════════════════════════════════"
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

main() {
    print_banner

    # Check current system
    print_step "Checking current system..."
    echo "Current kernel: $(uname -r)"
    echo "Current mount: $(mount | grep ' / ' | awk '{print $1}')"

    # Check if ZFS is available
    if ! command -v zfs >/dev/null 2>&1; then
        print_error "ZFS not available"
        print_warning "You may be in LiveCD. Install ZFS first:"
        echo "  sudo apt install zfsutils-linux"
        exit 1
    fi

    # Check if pool is imported
    print_step "Checking ZFS pool..."
    if ! echo "$SUDO_PASS" | sudo -S zpool list "$POOL" >/dev/null 2>&1; then
        print_warning "Pool not imported. Importing..."
        echo "$SUDO_PASS" | sudo -S zpool import "$POOL"
    fi

    # Load encryption key if needed
    if ! echo "$SUDO_PASS" | sudo -S zfs get keystatus "$POOL" | grep -q "available"; then
        print_step "Loading encryption key..."
        echo "$ZFS_PASS" | echo "$SUDO_PASS" | sudo -S zfs load-key "$POOL"
    fi

    print_success "Pool ready: $POOL"

    # List boot environments
    print_step "Current boot environments:"
    echo "$SUDO_PASS" | sudo -S zfs list -r "$POOL/ROOT" -o name,used,refer,mountpoint | grep -E "NAME|ROOT/"

    # Check if livecd-xen-ai exists
    if echo "$SUDO_PASS" | sudo -S zfs list "$POOL/ROOT/$BE_NAME" >/dev/null 2>&1; then
        print_success "Boot environment exists: $BE_NAME"
    else
        print_error "Boot environment MISSING: $BE_NAME"
        print_warning "It may have been destroyed or rolled back"
        echo ""
        echo "Available boot environments:"
        echo "$SUDO_PASS" | sudo -S zfs list -r "$POOL/ROOT" -d 1
        exit 1
    fi

    # Check current bootfs
    print_step "Checking current bootfs setting..."
    CURRENT_BOOTFS=$(echo "$SUDO_PASS" | sudo -S zpool get bootfs "$POOL" -H -o value)
    echo "Current bootfs: $CURRENT_BOOTFS"

    if [ "$CURRENT_BOOTFS" = "$POOL/ROOT/$BE_NAME" ]; then
        print_success "Bootfs already set to $BE_NAME"
    else
        print_warning "Bootfs is: $CURRENT_BOOTFS"
        echo ""
        read -p "Set bootfs to $POOL/ROOT/$BE_NAME? [Y/n]: " -r
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "$SUDO_PASS" | sudo -S zpool set bootfs="$POOL/ROOT/$BE_NAME" "$POOL"
            print_success "Bootfs updated to $BE_NAME"
        fi
    fi

    # Check ZFSBootMenu
    print_step "Checking ZFSBootMenu installation..."
    if [ -f "/boot/efi/EFI/zbm-recovery.efi" ]; then
        ZBM_SIZE=$(ls -lh /boot/efi/EFI/zbm-recovery.efi | awk '{print $5}')
        print_success "ZFSBootMenu present ($ZBM_SIZE)"
    else
        print_error "ZFSBootMenu NOT found at /boot/efi/EFI/zbm-recovery.efi"
        print_warning "Reinstalling ZFSBootMenu..."

        # Reinstall ZFSBootMenu
        echo "$SUDO_PASS" | sudo -S mkdir -p /boot/efi/EFI
        echo "$SUDO_PASS" | sudo -S wget -O /boot/efi/EFI/zbm-recovery.efi \
            https://get.zfsbootmenu.org/efi/recovery

        # Create UEFI entry if missing
        if ! efibootmgr | grep -q "ZFSBootMenu"; then
            echo "$SUDO_PASS" | sudo -S efibootmgr --create --disk /dev/nvme0n1 --part 1 \
                --label "ZFSBootMenu-Xen" \
                --loader '\EFI\zbm-recovery.efi'
            print_success "UEFI entry created"
        fi
    fi

    # Check EFI boot entries
    print_step "Checking UEFI boot entries..."
    efibootmgr | grep -E "Boot|ZFS|debian"

    # Check if kernel exists in BE
    print_step "Checking kernel in boot environment..."
    if echo "$SUDO_PASS" | sudo -S zfs mount "$POOL/ROOT/$BE_NAME" 2>/dev/null; then
        MOUNT_POINT=$(echo "$SUDO_PASS" | sudo -S zfs get mountpoint "$POOL/ROOT/$BE_NAME" -H -o value)

        if [ -f "$MOUNT_POINT/boot/vmlinuz-6.16.12-xen-ai-hardened" ]; then
            print_success "Kernel found in $BE_NAME"
        else
            print_error "Kernel MISSING in $BE_NAME"
            echo "Available kernels:"
            ls -lh "$MOUNT_POINT/boot/vmlinuz-"* 2>/dev/null || echo "  None found"
        fi

        echo "$SUDO_PASS" | sudo -S zfs unmount "$POOL/ROOT/$BE_NAME" 2>/dev/null || true
    fi

    # Final instructions
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  RECOVERY ACTIONS COMPLETE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}\n"

    echo -e "${CYAN}Current Status:${NC}"
    echo "  Pool: $POOL (imported)"
    echo "  Bootfs: $CURRENT_BOOTFS"
    echo "  BE exists: $BE_NAME"
    echo "  ZFSBootMenu: $([ -f /boot/efi/EFI/zbm-recovery.efi ] && echo 'Installed' || echo 'Missing')"
    echo ""

    echo -e "${CYAN}To boot into $BE_NAME:${NC}"
    echo "  1. Reboot: sudo reboot"
    echo "  2. At ZFSBootMenu, enter password: $ZFS_PASS"
    echo "  3. Select: $BE_NAME from menu"
    echo "  4. Press Enter"
    echo ""

    echo -e "${CYAN}If $BE_NAME doesn't appear in menu:${NC}"
    echo "  1. At ZFSBootMenu, press X for recovery shell"
    echo "  2. Run: zpool import $POOL"
    echo "  3. Run: echo '$ZFS_PASS' | zfs load-key $POOL"
    echo "  4. Run: zpool set bootfs=$POOL/ROOT/$BE_NAME $POOL"
    echo "  5. Run: exit"
    echo "  6. Select $BE_NAME from menu"
    echo ""

    echo -e "${CYAN}If system froze during last boot:${NC}"
    echo "  Option 1: Boot into LONENOMAD_NEW_ROLL (your original system)"
    echo "  Option 2: Check Xen/kernel logs for errors"
    echo "  Option 3: Edit kernel command line (press K in ZFSBootMenu)"
    echo "            Remove 'quiet splash' to see boot messages"
    echo ""

    echo -e "${YELLOW}Rollback if needed:${NC}"
    echo "  Boot into LONENOMAD_NEW_ROLL and run:"
    echo "  sudo zpool set bootfs=$POOL/ROOT/LONENOMAD_NEW_ROLL $POOL"
    echo ""
}

main "$@"
