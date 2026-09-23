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

--build-dir <DIR>            Path to the directory to use for building.

--install-dir <DIR>          Path to the directory to install the sysroot to.

--target <TRIPLE>            WASI target triple to build the sysroot for, for
                             example wasm32-wasip1.

--source-dir <DIR>           Path to an existing wasi-libc checkout to build.
                             When omitted, a known good version is downloaded.
EOF
}

while [[ $# -gt 0 ]]; do
    case ${1} in
        -h|--help)
            usage
            exit 0
            ;;
        --build-dir)
            build_dir="${2}"
            shift; shift
            ;;
        --install-dir)
            install_dir="${2}"
            shift; shift
            ;;
        --target)
            target="${2}"
            shift; shift
            ;;
        --source-dir)
            source_dir="${2}"
            shift; shift
            ;;
        *)
            error "Unknown argument '${1}'"
            ;;
    esac
done

for arg in build_dir install_dir target; do
    if [ -z ${!arg+x} ]; then
        error "Missing required argument '--${arg//_/-}'"
    elif [ "${!arg}" == "" ]; then
        error "Argument to --${arg//_/-} must not be empty"
    fi
done

CMAKE="${CMAKE:-cmake}"
wasi_libc_build_dir="${build_dir}/wasi-libc-build"

if [ -z ${source_dir+x} ]; then
    echo "--- Downloading wasi-libc"
    source_dir="${build_dir}/wasi-libc-source"
    mkdir -p "${source_dir}"
    wasi_libc_commit="2fc32bc81b9f07f8d9525edea59bfbaf760c06d6"
    curl -L "https://github.com/WebAssembly/wasi-libc/archive/${wasi_libc_commit}.zip" --output "${source_dir}/wasi-libc.zip"
    unzip -q "${source_dir}/wasi-libc.zip" -d "${source_dir}"
    mv "${source_dir}/wasi-libc-${wasi_libc_commit}"/* "${source_dir}"
    rm -rf "${source_dir}/wasi-libc-${wasi_libc_commit}"
fi

echo "--- Building wasi-libc"
# CMake's compiler probe would link an executable, which needs the sysroot this
# script is about to create.
#
# Shared libraries are left out because wasi-libc downloads a prebuilt
# compiler-rt from wasi-sdk to link libc.so with.
${CMAKE} \
  -S "${source_dir}" \
  -B "${wasi_libc_build_dir}" \
  -GNinja \
  ${NINJA:+-DCMAKE_MAKE_PROGRAM=${NINJA}} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC:-cc}" \
  -DCMAKE_C_COMPILER_TARGET="${target}" \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_INSTALL_PREFIX="${install_dir}" \
  -DTARGET_TRIPLE="${target}" \
  -DBUILD_SHARED=OFF \
  -DBUILD_TESTS=OFF

${CMAKE} --build "${wasi_libc_build_dir}"
${CMAKE} --install "${wasi_libc_build_dir}"
