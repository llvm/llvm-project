#!/bin/bash
#
# Deploy LAT5150 Tactical Interface Shortcuts to Xen VMs
# Installs desktop shortcuts and autostart configuration
#
# Usage:
#   ./deploy-vm-shortcuts.sh <vm-ip> [vm-ip2] [vm-ip3] ...
#   ./deploy-vm-shortcuts.sh all  # Deploy to all running VMs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_DESKTOP_DIR="${SCRIPT_DIR}/xen-vm-desktop"

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

get_running_vms() {
    # Get IPs of running Xen VMs from xl list
    # This is a placeholder - customize based on your VM naming/IP scheme
    echo "192.168.100.10 192.168.100.11 192.168.100.12"
}

deploy_to_vm() {
    local VM_IP="$1"
    local SSH_USER="${SSH_USER:-root}"

    log_section "Deploying to VM: ${VM_IP}"

    # Test SSH connection
    log_info "Testing SSH connection..."
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${SSH_USER}@${VM_IP}" exit 2>/dev/null; then
        log_error "Cannot connect to ${VM_IP}"
        log_warn "Ensure SSH keys are configured: ssh-copy-id ${SSH_USER}@${VM_IP}"
        return 1
    fi

    # Copy autostart script
    log_info "Installing autostart script..."
    scp "${VM_DESKTOP_DIR}/tactical-tunnel-autostart.sh" \
        "${SSH_USER}@${VM_IP}:/tmp/" > /dev/null 2>&1

    ssh "${SSH_USER}@${VM_IP}" << 'EOF'
        # Install autostart script
        sudo mv /tmp/tactical-tunnel-autostart.sh /usr/local/bin/
        sudo chmod +x /usr/local/bin/tactical-tunnel-autostart.sh
        sudo chown root:root /usr/local/bin/tactical-tunnel-autostart.sh

        # Create log file
        sudo touch /var/log/tactical-tunnel.log
        sudo chmod 666 /var/log/tactical-tunnel.log

        echo "Autostart script installed"
EOF

    # Copy desktop shortcut
    log_info "Installing desktop shortcut..."
    scp "${VM_DESKTOP_DIR}/LAT5150-Tactical.desktop" \
        "${SSH_USER}@${VM_IP}:/tmp/" > /dev/null 2>&1

    ssh "${SSH_USER}@${VM_IP}" << 'EOF'
        # Install for all users
        sudo cp /tmp/LAT5150-Tactical.desktop /usr/share/applications/
        sudo chmod 644 /usr/share/applications/LAT5150-Tactical.desktop

        # Install for current user's desktop
        mkdir -p ~/Desktop
        cp /tmp/LAT5150-Tactical.desktop ~/Desktop/
        chmod +x ~/Desktop/LAT5150-Tactical.desktop

        # Make trusted on GNOME
        if command -v gio > /dev/null 2>&1; then
            gio set ~/Desktop/LAT5150-Tactical.desktop "metadata::trusted" yes
        fi

        echo "Desktop shortcut installed"
EOF

    # Copy autostart configuration
    log_info "Installing autostart configuration..."
    scp "${VM_DESKTOP_DIR}/tactical-autostart.desktop" \
        "${SSH_USER}@${VM_IP}:/tmp/" > /dev/null 2>&1

    ssh "${SSH_USER}@${VM_IP}" << 'EOF'
        # Install for current user
        mkdir -p ~/.config/autostart
        cp /tmp/tactical-autostart.desktop ~/.config/autostart/
        chmod 644 ~/.config/autostart/tactical-autostart.desktop

        # For all users (optional)
        sudo mkdir -p /etc/xdg/autostart
        sudo cp /tmp/tactical-autostart.desktop /etc/xdg/autostart/
        sudo chmod 644 /etc/xdg/autostart/tactical-autostart.desktop

        echo "Autostart configuration installed"
EOF

    # Setup SSH keys if not already configured
    log_info "Configuring SSH access to host..."
    ssh "${SSH_USER}@${VM_IP}" << 'EOF'
        # Generate SSH key if doesn't exist
        if [ ! -f ~/.ssh/id_ed25519 ]; then
            ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "tactical-vm-$(hostname)"
            echo "SSH key generated"
        else
            echo "SSH key already exists"
        fi

        # Show public key for manual copying to host
        echo ""
        echo "=== SSH Public Key (copy to host) ==="
        cat ~/.ssh/id_ed25519.pub
        echo "======================================"
        echo ""
        echo "On host, run: ssh-copy-id -i ~/.ssh/id_ed25519.pub root@192.168.100.1"
EOF

    log_info "✓ Deployment complete for ${VM_IP}"
    echo ""
}

show_help() {
    cat <<EOF
Deploy LAT5150 Tactical Interface Shortcuts to Xen VMs

Usage:
    $0 <vm-ip> [vm-ip2] [vm-ip3] ...
    $0 all

Examples:
    # Deploy to specific VMs
    $0 192.168.100.10 192.168.100.11

    # Deploy to all running VMs
    $0 all

    # Different SSH user
    SSH_USER=tactical $0 192.168.100.10

Description:
    Installs LAT5150 Tactical Interface shortcuts and autostart
    configuration on Xen VMs. This includes:

    - Desktop launcher shortcut
    - Application menu entry
    - Auto-start SSH tunnel on boot
    - SSH key configuration

Requirements:
    - SSH access to VMs
    - VMs running Linux with desktop environment
    - Root or sudo access on VMs

Post-Installation:
    - Shortcuts appear in application menu and desktop
    - SSH tunnel starts automatically on VM boot
    - Access interface at: http://localhost:5001

EOF
}

# Main
if [ $# -eq 0 ]; then
    log_error "No VM specified"
    echo ""
    show_help
    exit 1
fi

if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Check required files exist
if [ ! -d "$VM_DESKTOP_DIR" ]; then
    log_error "VM desktop directory not found: $VM_DESKTOP_DIR"
    exit 1
fi

if [ ! -f "${VM_DESKTOP_DIR}/LAT5150-Tactical.desktop" ]; then
    log_error "Desktop file not found"
    exit 1
fi

log_section "LAT5150 Tactical Interface - VM Shortcut Deployment"
echo ""

# Deploy to VMs
if [ "$1" = "all" ]; then
    log_info "Deploying to all running VMs..."
    VMS=$(get_running_vms)

    for VM_IP in $VMS; do
        deploy_to_vm "$VM_IP"
    done
else
    # Deploy to specified VMs
    for VM_IP in "$@"; do
        deploy_to_vm "$VM_IP"
    done
fi

log_section "Deployment Complete"
echo ""
log_info "Next steps:"
log_info "1. On each VM, copy SSH public key to host:"
log_info "   ssh-copy-id root@192.168.100.1"
log_info ""
log_info "2. On host, ensure tactical interface is running:"
log_info "   python 03-web-interface/secured_self_coding_api.py --port 5001"
log_info ""
log_info "3. Reboot VMs to test autostart, or run manually:"
log_info "   /usr/local/bin/tactical-tunnel-autostart.sh"
log_info ""
log_info "4. Access interface from VM:"
log_info "   firefox http://localhost:5001"
echo ""
