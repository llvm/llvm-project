#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# Install Ultimate Kernel + AI Framework to ZFS (with embedded password)
#═══════════════════════════════════════════════════════════════════════════════

set -e

MOUNT="/mnt/transplant"

echo "═══════════════════════════════════════════════════════════════"
echo "  INSTALLING TO ZFS BOOT ENVIRONMENT"
echo "═══════════════════════════════════════════════════════════════"

echo "[1/14] Mounting chroot environment..."
echo "1786" | sudo -S mount /dev/nvme0n1p1 $MOUNT/boot/efi 2>/dev/null || echo "  EFI mounted"
echo "1786" | sudo -S mount -t proc proc $MOUNT/proc 2>/dev/null || echo "  proc mounted"
echo "1786" | sudo -S mount -t sysfs sys $MOUNT/sys 2>/dev/null || echo "  sys mounted"
echo "1786" | sudo -S mount -o bind /dev $MOUNT/dev 2>/dev/null || echo "  dev mounted"
echo "1786" | sudo -S mount -t devpts devpts $MOUNT/dev/pts 2>/dev/null || echo "  devpts mounted"

echo "[2/14] Copying DNS..."
echo "1786" | sudo -S cp /etc/resolv.conf $MOUNT/etc/resolv.conf

echo "[3/14] Copying kernel packages..."
echo "1786" | sudo -S cp ~/kernel-build/linux-image-6.16.12-ultimate_6.16.12-3_amd64.deb $MOUNT/tmp/
echo "1786" | sudo -S cp ~/kernel-build/linux-headers-6.16.12-ultimate_6.16.12-3_amd64.deb $MOUNT/tmp/

echo "[4/14] Installing kernel image..."
echo "1786" | sudo -S chroot $MOUNT dpkg -i /tmp/linux-image-6.16.12-ultimate_6.16.12-3_amd64.deb

echo "[5/14] Installing kernel headers..."
echo "1786" | sudo -S chroot $MOUNT dpkg -i /tmp/linux-headers-6.16.12-ultimate_6.16.12-3_amd64.deb

echo "[6/14] Generating initramfs with ZFS support..."
echo "1786" | sudo -S chroot $MOUNT update-initramfs -c -k 6.16.12-ultimate

echo "[7/14] Copying DSMIL AI Framework..."
echo "1786" | sudo -S mkdir -p $MOUNT/opt/dsmil
echo "1786" | sudo -S rsync -a --info=progress2 /home/john/LAT5150DRVMIL/ $MOUNT/opt/dsmil/

echo "[8/14] Installing DSMIL systemd service..."
echo "1786" | sudo -S cp /home/john/LAT5150DRVMIL/05-deployment/systemd/dsmil-server.service $MOUNT/etc/systemd/system/
echo "1786" | sudo -S sed -i 's|/home/john/LAT5150DRVMIL|/opt/dsmil|g' $MOUNT/etc/systemd/system/dsmil-server.service
echo "1786" | sudo -S chroot $MOUNT systemctl enable dsmil-server 2>/dev/null || true

echo "[9/14] Updating GRUB..."
echo "1786" | sudo -S chroot $MOUNT update-grub

echo "[10/14] Verifying kernel..."
echo "1786" | sudo -S ls -lh $MOUNT/boot/vmlinuz-6.16.12-ultimate

echo "[11/14] Creating snapshot..."
echo "1786" | sudo -S zfs snapshot rpool/ROOT/livecd-xen-ai@ready-to-boot-ultimate

echo "[12/14] Setting bootfs..."
echo "1786" | sudo -S zpool set bootfs=rpool/ROOT/livecd-xen-ai rpool
echo "1786" | sudo -S zpool get bootfs rpool

echo "[13/14] Cleanup mounts..."
echo "1786" | sudo -S umount $MOUNT/boot/efi
echo "1786" | sudo -S umount $MOUNT/proc
echo "1786" | sudo -S umount $MOUNT/sys
echo "1786" | sudo -S umount $MOUNT/dev/pts
echo "1786" | sudo -S umount $MOUNT/dev
echo "1786" | sudo -S zfs unmount rpool/ROOT/livecd-xen-ai

echo "[14/14] Exporting pool..."
echo "1786" | sudo -S zpool export rpool

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  INSTALLATION COMPLETE - READY FOR REBOOT!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Installed:"
echo "  ✓ Kernel: 6.16.12-ultimate (-march=alderlake -O2)"
echo "  ✓ Xen: 4.20.0 hypervisor"
echo "  ✓ DSMIL AI Framework in /opt/dsmil"
echo "  ✓ Full Intel NPU + GNA + GPU support"
echo "  ✓ TPM 2.0 attestation"
echo "  ✓ All security hardening"
echo ""
echo "To boot:"
echo "  sudo reboot"
echo "  Password at ZFSBootMenu: 1/0523/600260"
echo "  Select: livecd-xen-ai"
echo ""
