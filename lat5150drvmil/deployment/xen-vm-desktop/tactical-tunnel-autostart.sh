#!/bin/bash
#
# LAT5150 Tactical Tunnel Auto-Start Script
# Automatically establishes SSH tunnel to tactical interface on VM boot
#

# Configuration
HOST_IP="${TACTICAL_HOST:-192.168.100.1}"
LOCAL_PORT=5001
REMOTE_PORT=5001
SSH_USER="${SSH_USER:-root}"
LOGFILE="/var/log/tactical-tunnel.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

# Check if tunnel already running
if pgrep -f "ssh.*${HOST_IP}.*${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" > /dev/null; then
    log "Tactical tunnel already running"
    exit 0
fi

# Wait for network to be ready
log "Waiting for network connectivity..."
for i in {1..30}; do
    if ping -c 1 -W 1 "$HOST_IP" > /dev/null 2>&1; then
        log "Network ready, host reachable"
        break
    fi
    sleep 1
done

# Verify SSH connectivity
log "Testing SSH connection to $HOST_IP..."
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
     "${SSH_USER}@${HOST_IP}" exit 2>/dev/null; then
    log "ERROR: SSH connection failed"
    # Show notification to user
    if command -v notify-send > /dev/null; then
        notify-send -u critical "Tactical Interface" \
            "SSH tunnel failed. Setup SSH keys with:\nssh-copy-id ${SSH_USER}@${HOST_IP}"
    fi
    exit 1
fi

# Create SSH tunnel
log "Establishing SSH tunnel to tactical interface..."
ssh -f -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${SSH_USER}@${HOST_IP}" \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=no \
    2>> "$LOGFILE"

if [ $? -eq 0 ]; then
    log "Tactical tunnel established successfully"

    # Show success notification
    if command -v notify-send > /dev/null; then
        notify-send -u normal "Tactical Interface" \
            "SSH tunnel established\nAccess: http://localhost:${LOCAL_PORT}"
    fi
else
    log "ERROR: Failed to establish tunnel"
    if command -v notify-send > /dev/null; then
        notify-send -u critical "Tactical Interface" "Failed to establish SSH tunnel"
    fi
    exit 1
fi

# Wait a moment and verify tunnel is up
sleep 2
if pgrep -f "ssh.*${HOST_IP}.*${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" > /dev/null; then
    log "Tunnel verified and running"
else
    log "ERROR: Tunnel failed verification"
    exit 1
fi

exit 0
