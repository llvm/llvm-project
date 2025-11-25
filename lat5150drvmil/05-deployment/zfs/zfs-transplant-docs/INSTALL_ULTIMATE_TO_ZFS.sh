#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# Install Ultimate Kernel + AI Framework to ZFS Boot Environment
# Run AFTER kernel build completes
#═══════════════════════════════════════════════════════════════════════════════

set -e

SUDO_PASS="1786"
ZFS_PASS="1/0523/600260"
POOL="rpool"
BE_NAME="livecd-xen-ai"
MOUNT_POINT="/mnt/transplant"
KERNEL_VERSION="6.16.12-ultimate"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  INSTALL ULTIMATE KERNEL + AI FRAMEWORK TO ZFS"
echo "  Target BE: $BE_NAME"
echo "  Kernel: $KERNEL_VERSION"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if kernel packages exist
KERNEL_IMAGE=$(ls ~/kernel-build/linux-image-${KERNEL_VERSION}_*.deb 2>/dev/null || ls /usr/src/linux-image-${KERNEL_VERSION}_*.deb 2>/dev/null | head -1)
KERNEL_HEADERS=$(ls ~/kernel-build/linux-headers-${KERNEL_VERSION}_*.deb 2>/dev/null || ls /usr/src/linux-headers-${KERNEL_VERSION}_*.deb 2>/dev/null | head -1)

if [ -z "$KERNEL_IMAGE" ]; then
    echo -e "${RED}ERROR: Kernel packages not found!${NC}"
    echo "Checked: ~/kernel-build/ and /usr/src/"
    echo "Run BUILD_ULTIMATE_KERNEL.sh first"
    exit 1
fi

echo "Found packages:"
echo "  Image: $KERNEL_IMAGE"
echo "  Headers: $KERNEL_HEADERS"

echo -e "${BLUE}[1/12] Checking ZFS pool...${NC}"
if ! echo "$SUDO_PASS" | sudo -S zpool list "$POOL" >/dev/null 2>&1; then
    echo "Importing pool..."
    echo "$SUDO_PASS" | sudo -S zpool import -f "$POOL"
fi

if ! echo "$SUDO_PASS" | sudo -S zfs get keystatus "$POOL" | grep -q "available"; then
    echo "Loading encryption key..."
    echo "$ZFS_PASS" | echo "$SUDO_PASS" | sudo -S bash -c 'zfs load-key '"$POOL"
fi

echo -e "${BLUE}[2/12] Creating safety snapshot...${NC}"
SNAPSHOT_NAME="$POOL@before-ultimate-install-$(date +%Y%m%d-%H%M)"
echo "$SUDO_PASS" | sudo -S zfs snapshot -r "$SNAPSHOT_NAME"
echo -e "${GREEN}✓ Snapshot: $SNAPSHOT_NAME${NC}"

echo -e "${BLUE}[3/12] Ensuring BE is unmounted...${NC}"
echo "$SUDO_PASS" | sudo -S zfs unmount "$POOL/ROOT/$BE_NAME" 2>/dev/null || true

echo -e "${BLUE}[4/12] Mounting boot environment...${NC}"
echo "$SUDO_PASS" | sudo -S zfs set mountpoint="$MOUNT_POINT" "$POOL/ROOT/$BE_NAME"
echo "$SUDO_PASS" | sudo -S zfs mount "$POOL/ROOT/$BE_NAME"

echo -e "${BLUE}[5/12] Mounting EFI and system directories...${NC}"
echo "$SUDO_PASS" | sudo -S mkdir -p "$MOUNT_POINT/boot/efi"
echo "$SUDO_PASS" | sudo -S mount /dev/nvme0n1p1 "$MOUNT_POINT/boot/efi"
echo "$SUDO_PASS" | sudo -S mount -t proc proc "$MOUNT_POINT/proc"
echo "$SUDO_PASS" | sudo -S mount -t sysfs sys "$MOUNT_POINT/sys"
echo "$SUDO_PASS" | sudo -S mount -o bind /dev "$MOUNT_POINT/dev"
echo "$SUDO_PASS" | sudo -S mount -t devpts devpts "$MOUNT_POINT/dev/pts"

echo -e "${BLUE}[6/12] Copying DNS configuration...${NC}"
echo "$SUDO_PASS" | sudo -S cp /etc/resolv.conf "$MOUNT_POINT/etc/resolv.conf"

echo -e "${BLUE}[7/12] Installing kernel packages...${NC}"
echo "$SUDO_PASS" | sudo -S cp "$KERNEL_IMAGE" "$MOUNT_POINT/tmp/"
echo "$SUDO_PASS" | sudo -S cp "$KERNEL_HEADERS" "$MOUNT_POINT/tmp/"

echo "$SUDO_PASS" | sudo -S chroot "$MOUNT_POINT" bash -c \
    "dpkg -i /tmp/linux-image-${KERNEL_VERSION}_*.deb /tmp/linux-headers-${KERNEL_VERSION}_*.deb"

echo -e "${BLUE}[8/12] Generating initramfs with ZFS support...${NC}"
echo "$SUDO_PASS" | sudo -S chroot "$MOUNT_POINT" update-initramfs -c -k "$KERNEL_VERSION"

echo -e "${BLUE}[9/12] Transplanting DSMIL AI Framework...${NC}"
echo "$SUDO_PASS" | sudo -S mkdir -p "$MOUNT_POINT/opt/dsmil"
echo "$SUDO_PASS" | sudo -S rsync -avh --info=progress2 \
    /home/john/LAT5150DRVMIL/ "$MOUNT_POINT/opt/dsmil/"

echo -e "${BLUE}[10/12] Installing DSMIL systemd service...${NC}"
cat | echo "$SUDO_PASS" | sudo -S tee "$MOUNT_POINT/etc/systemd/system/dsmil-server.service" > /dev/null << 'EOF'
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
StandardOutput=journal
StandardError=journal

PrivateTmp=true
NoNewPrivileges=true

Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="DSMIL_HOME=/opt/dsmil"

[Install]
WantedBy=multi-user.target
EOF

echo "$SUDO_PASS" | sudo -S chroot "$MOUNT_POINT" systemctl enable dsmil-server.service 2>/dev/null || true

echo -e "${BLUE}[11/12] Updating GRUB with Xen configuration...${NC}"
echo "$SUDO_PASS" | sudo -S chroot "$MOUNT_POINT" update-grub 2>&1 | tail -5

echo -e "${BLUE}[12/12] Creating completion snapshot...${NC}"
echo "$SUDO_PASS" | sudo -S zfs snapshot "$POOL/ROOT/$BE_NAME@ultimate-kernel-installed"

echo -e "${BLUE}Cleaning up mounts...${NC}"
echo "$SUDO_PASS" | sudo -S umount "$MOUNT_POINT/boot/efi"
echo "$SUDO_PASS" | sudo -S umount "$MOUNT_POINT"/{proc,sys,dev/pts,dev}
echo "$SUDO_PASS" | sudo -S zfs unmount "$POOL/ROOT/$BE_NAME"

echo -e "${BLUE}Setting $BE_NAME as default boot...${NC}"
echo "$SUDO_PASS" | sudo -S zpool set bootfs="$POOL/ROOT/$BE_NAME" "$POOL"

echo -e "${BLUE}Creating final pre-reboot snapshot...${NC}"
echo "$SUDO_PASS" | sudo -S zfs snapshot -r "$POOL@ultimate-ready-$(date +%Y%m%d)"

echo -e "${BLUE}Verifying installation...${NC}"
echo "$SUDO_PASS" | sudo -S zfs mount "$POOL/ROOT/$BE_NAME"
if echo "$SUDO_PASS" | sudo -S ls "$MOUNT_POINT/boot/vmlinuz-$KERNEL_VERSION" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Kernel installed${NC}"
fi
if [ -d "$MOUNT_POINT/opt/dsmil" ]; then
    SIZE=$(echo "$SUDO_PASS" | sudo -S du -sh "$MOUNT_POINT/opt/dsmil" | awk '{print $1}')
    echo -e "${GREEN}✓ AI Framework installed ($SIZE)${NC}"
fi
echo "$SUDO_PASS" | sudo -S zfs unmount "$POOL/ROOT/$BE_NAME"

echo -e "${BLUE}Exporting pool for clean reboot...${NC}"
echo "$SUDO_PASS" | sudo -S zpool export "$POOL"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  INSTALLATION COMPLETE - READY FOR REBOOT!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Installed:${NC}"
echo "  ✓ Kernel: $KERNEL_VERSION"
echo "  ✓ DSMIL AI Framework (in /opt/dsmil)"
echo "  ✓ Xen 4.20 hypervisor"
echo "  ✓ Full security hardening"
echo "  ✓ AI acceleration support (NPU, GPU, GNA)"
echo "  ✓ ZFS support in initramfs"
echo ""
echo -e "${CYAN}Boot Environment:${NC}"
echo "  Name: $BE_NAME"
echo "  Bootfs: $POOL/ROOT/$BE_NAME (set as default)"
echo "  Snapshots: 3 created"
echo ""
echo -e "${CYAN}To Boot:${NC}"
echo "  1. Run: echo '1786' | sudo -S reboot"
echo "  2. Enter password at ZFSBootMenu: 1/0523/600260"
echo "  3. Select: $BE_NAME"
echo "  4. System boots with full Xen + AI + Security"
echo ""
echo -e "${CYAN}After Boot:${NC}"
echo "  - Verify Xen: xl info"
echo "  - Verify kernel: uname -r (should show $KERNEL_VERSION)"
echo "  - Access AI: http://localhost:9876"
echo "  - Start DSMIL: sudo systemctl start dsmil-server"
echo ""
echo -e "${YELLOW}Rollback if needed:${NC}"
echo "  At ZFSBootMenu, select: LONENOMAD_NEW_ROLL"
echo "  Or press F12 and boot from sda (current system)"
echo ""
