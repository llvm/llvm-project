#!/usr/bin/env bash
#
# Stage the locally archived Meteor Lake microcode (revision 0x1c) for early
# loading and register a dedicated GRUB entry that uses it. Run with sudo.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MICROCODE_SRC="${REPO_ROOT}/04-hardware/microcode/06-aa-04_0x1c.bin"
MICROCODE_DEST="/lib/firmware/intel-ucode/06-aa-04"
MICROCODE_IMG="/boot/intel-ucode-0x1c.img"
GRUB_SNIPPET="/etc/grub.d/09_avx512_microcode"
KERNEL_VERSION="${KERNEL_VERSION:-$(uname -r)}"
KERNEL_IMAGE="/boot/vmlinuz-${KERNEL_VERSION}"
INITRD_IMAGE="/boot/initrd.img-${KERNEL_VERSION}"

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)." >&2
    exit 1
fi

if [[ ! -f "${MICROCODE_SRC}" ]]; then
    echo "Missing microcode blob: ${MICROCODE_SRC}" >&2
    exit 1
fi

if [[ ! -f "${KERNEL_IMAGE}" || ! -f "${INITRD_IMAGE}" ]]; then
    echo "Kernel or initrd not found for version ${KERNEL_VERSION}." >&2
    exit 1
fi

ROOT_DEVICE="$(findmnt -no SOURCE /)"
ROOT_UUID="$(blkid -s UUID -o value "${ROOT_DEVICE}" 2>/dev/null || true)"
if [[ -n "${ROOT_UUID}" ]]; then
    ROOT_SPEC="UUID=${ROOT_UUID}"
else
    ROOT_SPEC="${ROOT_DEVICE}"
fi

CMDLINE_DEFAULT="$(awk -F\" '/^GRUB_CMDLINE_LINUX_DEFAULT=/ {print $2}' /etc/default/grub)"
CMDLINE_SANITIZED="$(tr ' ' '\n' <<<"${CMDLINE_DEFAULT}" | sed '/^dis_ucode_ldr$/d;/^microcode=no$/d' | xargs)"
if [[ -z "${CMDLINE_SANITIZED}" ]]; then
    CMDLINE_SANITIZED="quiet"
fi

echo "[1/4] Installing microcode blob into ${MICROCODE_DEST}"
install -m 0644 "${MICROCODE_SRC}" "${MICROCODE_DEST}"

echo "[2/4] Building early microcode image at ${MICROCODE_IMG}"
WORKDIR="$(mktemp -d)"
cleanup() {
    rm -rf "${WORKDIR}"
}
trap cleanup EXIT

mkdir -p "${WORKDIR}/kernel/x86/microcode"
install -m 0644 "${MICROCODE_SRC}" "${WORKDIR}/kernel/x86/microcode/GenuineIntel.bin"

(cd "${WORKDIR}" && find . -print | cpio -o -H newc) > "${MICROCODE_IMG}"
chown root:root "${MICROCODE_IMG}"
chmod 0644 "${MICROCODE_IMG}"

echo "[3/4] Writing custom GRUB entry to ${GRUB_SNIPPET}"
cat > "${GRUB_SNIPPET}" <<EOF
#!/bin/sh
exec tail -n +3 \$0
# Custom AVX-512 entry that forces microcode revision 0x1c.
menuentry 'Debian GNU/Linux (AVX-512 microcode 0x1c)' --class debian --class gnu-linux --class gnu --class os {
    load_video
    insmod gzio
    insmod part_gpt
    insmod ext2
    search --no-floppy --fs-uuid --set=root ${ROOT_UUID}
    linux ${KERNEL_IMAGE} root=${ROOT_SPEC} ro ${CMDLINE_SANITIZED}
    initrd ${MICROCODE_IMG} ${INITRD_IMAGE}
}
EOF
chmod 0755 "${GRUB_SNIPPET}"

echo "[4/4] Regenerating GRUB configuration"
update-grub

echo "Done. Select the \"AVX-512 microcode 0x1c\" entry in GRUB to boot with the older microcode."
