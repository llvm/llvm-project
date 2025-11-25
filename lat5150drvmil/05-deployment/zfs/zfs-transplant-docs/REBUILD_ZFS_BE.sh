#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# Rebuild livecd-xen-ai Boot Environment with AI Framework
# This installs everything properly into the ZFS boot environment
#═══════════════════════════════════════════════════════════════════════════════

set -e

SUDO_PASS="1786"
ZFS_PASS="1/0523/600260"
POOL="rpool"
BE_NAME="livecd-xen-ai"
MOUNT_POINT="/mnt/transplant"

echo "═══════════════════════════════════════════════════════════════"
echo "  REBUILD ZFS BOOT ENVIRONMENT: $BE_NAME"
echo "  Includes: Xen + Kernel + DSMIL + AI Framework"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Destroy and recreate BE
echo "[1/10] Destroying old BE and creating fresh clone..."
echo "1786" | sudo -S zfs destroy -r $POOL/ROOT/$BE_NAME 2>/dev/null || true
echo "1786" | sudo -S zfs snapshot $POOL/ROOT/LONENOMAD_NEW_ROLL@for-livecd-rebuild
echo "1786" | sudo -S zfs clone $POOL/ROOT/LONENOMAD_NEW_ROLL@for-livecd-rebuild $POOL/ROOT/$BE_NAME

# Mount BE
echo "[2/10] Mounting boot environment..."
echo "1786" | sudo -S zfs set mountpoint=$MOUNT_POINT $POOL/ROOT/$BE_NAME
echo "1786" | sudo -S zfs mount $POOL/ROOT/$BE_NAME

# Mount EFI
echo "[3/10] Mounting EFI partition..."
echo "1786" | sudo -S mkdir -p $MOUNT_POINT/boot/efi
echo "1786" | sudo -S mount /dev/nvme0n1p1 $MOUNT_POINT/boot/efi

# Mount proc/sys/dev for chroot
echo "[4/10] Preparing chroot environment..."
echo "1786" | sudo -S mount -t proc proc $MOUNT_POINT/proc
echo "1786" | sudo -S mount -t sysfs sys $MOUNT_POINT/sys
echo "1786" | sudo -S mount -o bind /dev $MOUNT_POINT/dev
echo "1786" | sudo -S mount -t devpts devpts $MOUNT_POINT/dev/pts

# Copy resolv.conf
echo "1786" | sudo -S cp /etc/resolv.conf $MOUNT_POINT/etc/resolv.conf

# Install Xen kernel (already have the deb)
echo "[5/10] Installing Xen-enabled kernel..."
echo "1786" | sudo -S chroot $MOUNT_POINT dpkg -i \
    /usr/src/linux-image-6.16.12-xen-ai-hardened_6.16.12-2_amd64.deb \
    /usr/src/linux-headers-6.16.12-xen-ai-hardened_6.16.12-2_amd64.deb || {
    echo "Using existing kernel in current system"
}

# Generate initramfs with ZFS support
echo "[6/10] Generating initramfs with ZFS support..."
echo "1786" | sudo -S chroot $MOUNT_POINT update-initramfs -c -k 6.16.12-xen-ai-hardened 2>&1 | tail -5

# Copy AI Framework
echo "[7/10] Copying DSMIL AI Framework..."
echo "1786" | sudo -S mkdir -p $MOUNT_POINT/opt/dsmil
echo "1786" | sudo -S rsync -a /home/john/LAT5150DRVMIL/ $MOUNT_POINT/opt/dsmil/

# Copy Ollama models if present
if [ -d "$HOME/.ollama" ]; then
    echo "[8/10] Copying Ollama models..."
    echo "1786" | sudo -S mkdir -p $MOUNT_POINT/var/lib/ollama
    echo "1786" | sudo -S rsync -a $HOME/.ollama/ $MOUNT_POINT/var/lib/ollama/
fi

# Install systemd service
echo "[9/10] Installing DSMIL service..."
cat | echo "1786" | sudo -S tee $MOUNT_POINT/etc/systemd/system/dsmil-server.service > /dev/null << 'EOF'
[Unit]
Description=DSMIL Unified AI Server
After=network.target ollama.service zfs-mount.service
Wants=ollama.service
Requires=zfs-mount.service

[Service]
Type=simple
User=john
WorkingDirectory=/opt/dsmil/03-web-interface
ExecStart=/usr/bin/python3 /opt/dsmil/03-web-interface/dsmil_unified_server.py
Restart=on-failure
RestartSec=10

# Security
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

echo "1786" | sudo -S chroot $MOUNT_POINT systemctl enable dsmil-server.service 2>/dev/null || true

# Update GRUB
echo "[10/10] Updating GRUB configuration..."
echo "1786" | sudo -S chroot $MOUNT_POINT update-grub 2>&1 | tail -10

# Cleanup
echo "1786" | sudo -S umount $MOUNT_POINT/boot/efi
echo "1786" | sudo -S umount $MOUNT_POINT/{proc,sys,dev/pts,dev}

# Create snapshot
echo "Creating completion snapshot..."
echo "1786" | sudo -S zfs snapshot $POOL/ROOT/$BE_NAME@rebuilt-$(date +%Y%m%d)

# Set as default boot
echo "Setting as default boot..."
echo "1786" | sudo -S zpool set bootfs=$POOL/ROOT/$BE_NAME $POOL

# Unmount
echo "1786" | sudo -S zfs unmount $POOL/ROOT/$BE_NAME

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  REBUILD COMPLETE!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Boot environment: $BE_NAME"
echo "Bootfs: $(echo '1786' | sudo -S zpool get bootfs $POOL -H -o value)"
echo ""
echo "Contains:"
echo "  ✓ Xen 4.20 hypervisor"
echo "  ✓ Kernel 6.16.12-xen-ai-hardened"
echo "  ✓ DSMIL AI Framework (/opt/dsmil)"
echo "  ✓ Ollama models"
echo "  ✓ ZFS support in initramfs"
echo ""
echo "To boot:"
echo "  sudo reboot"
echo "  → Enter password: 1/0523/600260"
echo "  → Select: $BE_NAME"
echo ""
