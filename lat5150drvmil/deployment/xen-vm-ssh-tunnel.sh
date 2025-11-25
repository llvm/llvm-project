#!/bin/bash
#
# SSH Tunnel Script for Xen VM Access to Tactical Interface
# Run this script INSIDE a Xen VM to create secure tunnel to host
#
# Usage (from within VM):
#   ./xen-vm-ssh-tunnel.sh <host-ip> [local-port] [remote-port]
#
# Example:
#   ./xen-vm-ssh-tunnel.sh 192.168.100.1 5001 5001
#   # Then access: http://localhost:5001 in VM browser
#

set -e

HOST_IP="${1}"
LOCAL_PORT="${2:-5001}"
REMOTE_PORT="${3:-5001}"
SSH_USER="${SSH_USER:-root}"

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

if [ -z "$HOST_IP" ]; then
    cat <<EOF
${RED}ERROR:${NC} Host IP address required

${CYAN}Usage:${NC}
    $0 <host-ip> [local-port] [remote-port]

${CYAN}Examples:${NC}
    # Basic tunnel (default ports)
    $0 192.168.100.1

    # Custom ports
    $0 192.168.100.1 8080 5001

    # Different SSH user
    SSH_USER=tactical $0 192.168.100.1

${CYAN}Description:${NC}
    Creates SSH tunnel from VM to host tactical interface
    VM will access interface at: http://localhost:${LOCAL_PORT}

${CYAN}Requirements:${NC}
    - SSH access to host
    - Host running tactical interface on port ${REMOTE_PORT}
    - SSH key-based authentication recommended

${CYAN}First Time Setup:${NC}
    1. Generate SSH key: ssh-keygen -t ed25519
    2. Copy to host: ssh-copy-id ${SSH_USER}@<host-ip>
    3. Run this script

EOF
    exit 1
fi

log_section "Xen VM SSH Tunnel to Tactical Interface"
echo ""
log_info "Host: ${HOST_IP}"
log_info "Local Port: ${LOCAL_PORT}"
log_info "Remote Port: ${REMOTE_PORT}"
log_info "SSH User: ${SSH_USER}"
echo ""

# Check SSH connection
log_info "Testing SSH connection..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "${SSH_USER}@${HOST_IP}" exit 2>/dev/null; then
    log_info "✓ SSH connection successful"
else
    log_error "✗ SSH connection failed"
    echo ""
    log_warn "Setup SSH key-based authentication:"
    log_warn "  1. ssh-keygen -t ed25519"
    log_warn "  2. ssh-copy-id ${SSH_USER}@${HOST_IP}"
    log_warn "  3. Try again"
    exit 1
fi

# Check if port already in use
if ss -ltn | grep -q ":${LOCAL_PORT} "; then
    log_warn "Port ${LOCAL_PORT} already in use locally"

    # Check if it's our tunnel
    if pgrep -f "ssh.*${HOST_IP}.*${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" > /dev/null; then
        log_info "Existing tunnel found, killing..."
        pkill -f "ssh.*${HOST_IP}.*${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" || true
        sleep 1
    else
        log_error "Port ${LOCAL_PORT} in use by another process"
        log_warn "Choose different local port or stop the process"
        exit 1
    fi
fi

# Create SSH tunnel
log_info "Creating SSH tunnel..."
echo ""

ssh -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${SSH_USER}@${HOST_IP}" \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=no &

SSH_PID=$!

# Wait for tunnel to establish
sleep 2

# Check if tunnel is up
if kill -0 $SSH_PID 2>/dev/null; then
    log_info "✓ SSH tunnel established (PID: $SSH_PID)"
    echo ""
    log_section "Tunnel Active"
    echo ""
    log_info "Access tactical interface at:"
    log_info "  ${GREEN}http://localhost:${LOCAL_PORT}${NC}"
    echo ""
    log_info "To stop tunnel:"
    log_info "  kill $SSH_PID"
    log_info "  OR: pkill -f 'ssh.*${HOST_IP}.*${LOCAL_PORT}'"
    echo ""
    log_warn "Keep this terminal open to maintain tunnel"
    echo ""

    # Wait for tunnel
    wait $SSH_PID
else
    log_error "✗ Failed to establish SSH tunnel"
    exit 1
fi
