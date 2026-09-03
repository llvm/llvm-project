#!/usr/bin/env bash
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

set -e

PROGNAME="$(basename "${0}")"

function error() { printf "error: %s\n" "$*" >&2; exit 1; }

function usage() {
cat <<EOF
Usage:
${PROGNAME} [options]

[-h|--help]                  Display this help and exit.

--install-dir <DIR>          Path to the directory to install the runtime to.
                             The runtime is installed as <DIR>/wasmkit.
EOF
}

while [[ $# -gt 0 ]]; do
    case ${1} in
        -h|--help)
            usage
            exit 0
            ;;
        --install-dir)
            install_dir="${2}"
            shift; shift
            ;;
        *)
            error "Unknown argument '${1}'"
            ;;
    esac
done

if [ -z ${install_dir+x} ]; then
    error "Missing required argument '--install-dir'"
elif [ "${install_dir}" == "" ]; then
    error "Argument to --install-dir must not be empty"
fi

wasmkit_version="0.3.1"

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)  asset="wasmkit-arm64-apple-macos" ;;
    Darwin-x86_64) asset="wasmkit-x86_64-apple-macos" ;;
    Linux-aarch64) asset="wasmkit-aarch64-swift-linux-musl" ;;
    Linux-x86_64)  asset="wasmkit-x86_64-swift-linux-musl" ;;
    *)             error "No WasmKit release is published for $(uname -s) $(uname -m)" ;;
esac

echo "--- Downloading WasmKit"
mkdir -p "${install_dir}"
curl -L "https://github.com/swiftwasm/WasmKit/releases/download/${wasmkit_version}/${asset}.tar.gz" \
    --output "${install_dir}/wasmkit.tar.gz"
tar -xzf "${install_dir}/wasmkit.tar.gz" -C "${install_dir}" --strip-components=1
