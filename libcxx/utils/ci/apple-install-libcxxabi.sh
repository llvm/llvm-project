#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

set -e

PROGNAME="$(basename "${0}")"

function error() { printf "error: %s\n" "$*"; exit 1; }

function usage() {
cat <<EOF
Usage:
${PROGNAME} [options]

[-h|--help]                  Display this help and exit.

--llvm-root <DIR>            Full path to the root of the LLVM monorepo. Only the libcxx
                             and libcxxabi directories are required.

--build-dir <DIR>            Full path to the directory to use for building. This will
                             contain intermediate build products.

--install-dir <DIR>          Full path to the directory to install the library to.

--symbols-dir <DIR>          Full path to the directory to install the .dSYM bundle to.

--sdk <SDK>                  SDK used for building the library. This represents
                             the target platform that the library will run on.
                             You can get a list of SDKs with \`xcodebuild -showsdks\`.

--architectures "<arch>..."  A whitespace separated list of architectures to build for.
                             The library will be built for each architecture independently,
                             and a universal binary containing all architectures will be
                             created from that.

--version X[.Y[.Z]]          The version of the library to encode in the dylib.

--cache <PATH>               The CMake cache to use to control how the library gets built.
EOF
}

while [[ $# -gt 0 ]]; do
    case ${1} in
        -h|--help)
            usage
            exit 0
            ;;
        --llvm-root)
            llvm_root="${2}"
            shift; shift
            ;;
        --build-dir)
            build_dir="${2}"
            shift; shift
            ;;
        --symbols-dir)
            symbols_dir="${2}"
            shift; shift
            ;;
        --install-dir)
            install_dir="${2}"
            shift; shift
            ;;
        --sdk)
            sdk="${2}"
            shift; shift
            ;;
        --architectures)
            architectures="${2}"
            shift; shift
            ;;
        --version)
            version="${2}"
            shift; shift
            ;;
        --cache)
            cache="${2}"
            shift; shift
            ;;
        *)
            error "Unknown argument '${1}'"
            ;;
    esac
done

for arg in llvm_root build_dir symbols_dir install_dir sdk architectures version cache; do
    if [ -z ${!arg+x} ]; then
        error "Missing required argument '--${arg//_/-}'"
    elif [ "${!arg}" == "" ]; then
        error "Argument to --${arg//_/-} must not be empty"
    fi
done

function step() {
    separator="$(printf "%0.s-" $(seq 1 ${#1}))"
    echo
    echo "${separator}"
    echo "${1}"
    echo "${separator}"
}

install_name_dir="/usr/lib"
headers_prefix="${install_dir}"

for arch in ${architectures}; do
    step "Building libc++abi.dylib for architecture ${arch}"
    mkdir -p "${build_dir}/${arch}"
    (cd "${build_dir}/${arch}" &&
        xcrun --sdk "${sdk}" cmake "${llvm_root}/llvm" \
            -GNinja \
            -DCMAKE_MAKE_PROGRAM="$(xcrun --sdk "${sdk}" --find ninja)" \
            -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" \
            -C "${cache}" \
            -DCMAKE_INSTALL_PREFIX="${build_dir}/${arch}-install" \
            -DCMAKE_INSTALL_NAME_DIR="${install_name_dir}" \
            -DCMAKE_OSX_ARCHITECTURES="${arch}" \
            -DLIBCXXABI_LIBRARY_VERSION="${version}" \
            -DLIBCXXABI_LIBCXX_PATH="${llvm_root}/libcxx"
    )

    xcrun --sdk "${sdk}" cmake --build "${build_dir}/${arch}" --target install-cxxabi -- -v
done

all_dylibs=$(for arch in ${architectures}; do
    echo "${build_dir}/${arch}-install/lib/libc++abi.dylib"
done)

all_archives=$(for arch in ${architectures}; do
    echo "${build_dir}/${arch}-install/lib/libc++abi.a"
done)

step "Creating a universal dylib from the dylibs for each architecture at ${install_dir}/usr/lib"
xcrun --sdk "${sdk}" lipo -create ${all_dylibs} -output "${build_dir}/libc++abi.dylib"

step "Installing the (stripped) universal dylib to ${install_dir}/usr/lib"
mkdir -p "${install_dir}/usr/lib"
cp "${build_dir}/libc++abi.dylib" "${install_dir}/usr/lib/libc++abi.dylib"
xcrun --sdk "${sdk}" strip -S "${install_dir}/usr/lib/libc++abi.dylib"

step "Installing the unstripped dylib and the dSYM bundle to ${symbols_dir}"
xcrun --sdk "${sdk}" dsymutil "${build_dir}/libc++abi.dylib" -o "${symbols_dir}/libc++abi.dylib.dSYM"
cp "${build_dir}/libc++abi.dylib" "${symbols_dir}/libc++abi.dylib"

step "Creating a universal static archive from the static archives for each architecture"
mkdir -p "${install_dir}/usr/local/lib/libcxx"
xcrun --sdk "${sdk}" libtool -static ${all_archives} -o "${install_dir}/usr/local/lib/libcxx/libc++abi-static.a"

#
# Install the headers by copying the headers from the source directory into
# the install directory.
# TODO: In the future, we should install the headers through CMake.
#
step "Installing the libc++abi headers to ${headers_prefix}/usr/include"
mkdir -p "${headers_prefix}/usr/include"
ditto "${llvm_root}/libcxxabi/include" "${headers_prefix}/usr/include"
if [[ $EUID -eq 0 ]]; then # Only chown if we're running as root
    chown -R root:wheel "${headers_prefix}/usr/include"
fi

step "Installing the libc++abi license"
mkdir -p "${headers_prefix}/usr/local/OpenSourceLicenses"
cp "${llvm_root}/libcxxabi/LICENSE.TXT" "${headers_prefix}/usr/local/OpenSourceLicenses/libcxxabi.txt"
