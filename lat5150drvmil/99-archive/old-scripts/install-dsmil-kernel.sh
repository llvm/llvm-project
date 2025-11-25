#!/bin/bash
# Install DSMIL Kernel with AVX-512 Enabler
# Dell Latitude 5450 MIL-SPEC with Mode 5 Platform Integrity

set -e

KERNEL_VERSION="6.16.9-dsmil"
KERNEL_SRC="/home/john/linux-6.16.9"
BUILD_DIR="${KERNEL_SRC}"

echo "═══════════════════════════════════════════════════════════"
echo "  DSMIL KERNEL INSTALLATION - AVX-512 UNLOCK"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Kernel Version: ${KERNEL_VERSION}"
echo "Build Directory: ${BUILD_DIR}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Must run as root (use sudo)"
    exit 1
fi

# Verify kernel is built
if [ ! -f "${BUILD_DIR}/arch/x86/boot/bzImage" ]; then
    echo "ERROR: Kernel not built at ${BUILD_DIR}/arch/x86/boot/bzImage"
    exit 1
fi

echo "[1/6] Installing kernel image..."
cp -v "${BUILD_DIR}/arch/x86/boot/bzImage" "/boot/vmlinuz-${KERNEL_VERSION}"

echo ""
echo "[2/6] Installing kernel modules..."
cd "${BUILD_DIR}"
make modules_install INSTALL_MOD_PATH=/ KERNELRELEASE="${KERNEL_VERSION}"

echo ""
echo "[3/6] Installing DSMIL AVX-512 enabler module..."
mkdir -p "/lib/modules/${KERNEL_VERSION}/extra"
cp -v /home/john/livecd-gen/kernel-modules/dsmil_avx512_enabler.ko \
    "/lib/modules/${KERNEL_VERSION}/extra/"

echo ""
echo "[4/6] Running depmod..."
depmod -a "${KERNEL_VERSION}"

echo ""
echo "[5/6] Creating initramfs..."
if command -v update-initramfs >/dev/null 2>&1; then
    update-initramfs -c -k "${KERNEL_VERSION}"
elif command -v dracut >/dev/null 2>&1; then
    dracut --force --hostonly "/boot/initrd.img-${KERNEL_VERSION}" "${KERNEL_VERSION}"
else
    echo "WARNING: No initramfs tool found (update-initramfs or dracut)"
    echo "You may need to create initramfs manually"
fi

echo ""
echo "[6/6] Updating bootloader..."
if [ -f /etc/default/grub ]; then
    echo "Updating GRUB configuration..."
    update-grub || grub-mkconfig -o /boot/grub/grub.cfg
elif [ -d /boot/efi/loader ]; then
    echo "Creating systemd-boot entry..."
    cat > /boot/efi/loader/entries/dsmil-avx512.conf <<EOF
title   DSMIL Kernel ${KERNEL_VERSION} (AVX-512 Enabled)
linux   /vmlinuz-${KERNEL_VERSION}
initrd  /initrd.img-${KERNEL_VERSION}
options root=UUID=$(blkid -s UUID -o value $(df / | tail -1 | awk '{print $1}')) ro dis_ucode_ldr quiet
EOF
    echo "Entry created: /boot/efi/loader/entries/dsmil-avx512.conf"
else
    echo "WARNING: Could not detect bootloader type"
    echo "Manual bootloader configuration may be needed"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  INSTALLATION COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Installed Kernel: /boot/vmlinuz-${KERNEL_VERSION}"
echo "Installed Modules: /lib/modules/${KERNEL_VERSION}/"
echo "AVX-512 Module: /lib/modules/${KERNEL_VERSION}/extra/dsmil_avx512_enabler.ko"
echo ""
echo "NEXT STEPS:"
echo "1. Review bootloader configuration"
echo "2. Reboot into new kernel"
echo "3. After boot, load AVX-512 module:"
echo "   sudo modprobe dsmil_avx512_enabler"
echo "4. Verify AVX-512 unlock:"
echo "   cat /proc/dsmil_avx512"
echo "   cat /proc/cpuinfo | grep avx512"
echo ""
echo "READY TO REBOOT!"
echo "═══════════════════════════════════════════════════════════"
