#!/usr/bin/env bash
set -euo pipefail

MODULE_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/01-source/drivers/dsmil_avx512_enabler"
KDIR="/lib/modules/$(uname -r)/build"
INSTALL=false

usage() {
    cat <<'EOF'
Usage: scripts/build_dsmil_avx512_enabler.sh [--install]

Builds the DSMIL AVX-512 enabler kernel module using the in-tree Kbuild files.

    --install   Copy the resulting .ko to /lib/modules/$(uname -r)/extra and run depmod
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)
            INSTALL=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

echo "[+] Building dsmil_avx512_enabler against ${KDIR}"
make -C "${KDIR}" M="${MODULE_DIR}" modules

KO_PATH="${MODULE_DIR}/dsmil_avx512_enabler.ko"
if [[ ! -f "${KO_PATH}" ]]; then
    echo "Build finished but ${KO_PATH} is missing" >&2
    exit 1
fi

if [[ "${INSTALL}" == true ]]; then
    DEST="/lib/modules/$(uname -r)/extra/dsmil_avx512_enabler.ko"
    echo "[+] Installing ${KO_PATH} -> ${DEST}"
    sudo install -D -m 0644 "${KO_PATH}" "${DEST}"
    sudo depmod -a
    echo "[i] Module installed. Load it with: sudo modprobe dsmil_avx512_enabler"
else
    echo "[i] Module built at ${KO_PATH}"
    echo "[i] Load manually with: sudo insmod ${KO_PATH}"
fi
