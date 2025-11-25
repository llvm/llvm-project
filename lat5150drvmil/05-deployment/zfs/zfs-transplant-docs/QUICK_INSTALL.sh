#!/bin/bash
# Quick manual install - run each command step by step
# sudo password: 1786

MOUNT="/mnt/transplant"

echo "Step 1: Mount EFI and chroot dirs..."
sudo mount /dev/nvme0n1p1 $MOUNT/boot/efi 2>/dev/null || echo "EFI mounted"
sudo mount -t proc proc $MOUNT/proc 2>/dev/null || echo "proc mounted"
sudo mount -t sysfs sys $MOUNT/sys 2>/dev/null || echo "sys mounted"
sudo mount -o bind /dev $MOUNT/dev 2>/dev/null || echo "dev mounted"
sudo mount -t devpts devpts $MOUNT/dev/pts 2>/dev/null || echo "devpts mounted"

echo "Step 2: Copy kernel..."
sudo cp ~/kernel-build/linux-image-6.16.12-ultimate_6.16.12-3_amd64.deb $MOUNT/tmp/
sudo cp ~/kernel-build/linux-headers-6.16.12-ultimate_6.16.12-3_amd64.deb $MOUNT/tmp/

echo "Step 3: Install kernel..."
sudo chroot $MOUNT dpkg -i /tmp/linux-*ultimate*.deb

echo "Step 4: Generate initramfs..."
sudo chroot $MOUNT update-initramfs -c -k 6.16.12-ultimate

echo "Step 5: Copy AI framework..."
sudo mkdir -p $MOUNT/opt/dsmil
sudo rsync -a /home/john/LAT5150DRVMIL/ $MOUNT/opt/dsmil/

echo "Step 6: Install DSMIL service..."
sudo cp /home/john/LAT5150DRVMIL/05-deployment/systemd/dsmil-server.service $MOUNT/etc/systemd/system/
sudo sed -i 's|/home/john/LAT5150DRVMIL|/opt/dsmil|g' $MOUNT/etc/systemd/system/dsmil-server.service
sudo chroot $MOUNT systemctl enable dsmil-server

echo "Step 7: Update GRUB..."
sudo chroot $MOUNT update-grub

echo "Step 8: Set bootfs..."
sudo zpool set bootfs=rpool/ROOT/livecd-xen-ai rpool

echo "Step 9: Create snapshot..."
sudo zfs snapshot rpool/ROOT/livecd-xen-ai@ready-to-boot

echo "Step 10: Cleanup..."
sudo umount $MOUNT/boot/efi $MOUNT/proc $MOUNT/sys $MOUNT/dev/pts $MOUNT/dev
sudo zfs unmount rpool/ROOT/livecd-xen-ai

echo "Step 11: Export pool..."
sudo zpool export rpool

echo "COMPLETE! Ready to reboot!"
echo "Run: sudo reboot"
