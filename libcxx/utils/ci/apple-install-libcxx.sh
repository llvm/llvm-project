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
function usage() {
cat <<EOF
Usage:
${PROGNAME} [-h|--help] --llvm-root <DIR> --build-dir <DIR> --install-dir <DIR> --symbols-dir <DIR> --sdk <SDK> --architectures <architectures...> --version <X.Y.Z> --cache <PATH>

--llvm-root     Full path to the root of the LLVM monorepo. Only the libcxx
                and libcxxabi directories are required.

--build-dir     Directory to use for building. This will contain intermediate
                build products.

--install-dir   Directory to install the library to.

--symbols-dir   Directory to install the .dSYM bundle to.

--sdk           SDK used for building the library. This represents the target
                platform that the library will run on. You can get a list of
                SDKs with \`xcodebuild -showsdks\`.

--architectures A whitespace separated list of architectures to build for.
                The library will be built for each architecture independently,
                and a universal binary containing all architectures will be
                created from that.

--version       The version of the library to encode in the dylib.

--cache         The CMake cache to use to control how the library gets built.
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
            architectures=""
            shift
            while [[ $# -gt 0 ]]; do
                if [[ "${1}" == "-"* ]]; then
                    break
                fi
                architectures+=" ${1}"
                shift
            done
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
            echo "Unknown argument '${1}'"
            exit 1
            ;;
    esac
done

for arg in llvm_root build_dir symbols_dir install_dir sdk architectures version cache; do
    if [ -z ${!arg+x} ]; then
        echo "Missing required argument '--${arg//_/-}'"
        exit 1
    elif [ "${!arg}" == "" ]; then
        echo "Argument to --${arg//_/-} must not be empty"
        exit 1
    fi
done

install_name_dir="/usr/lib"
dylib_name="libc++.1.dylib"
make_symlink="yes"

function step() {
    separator="$(printf "%0.s-" $(seq 1 ${#1}))"
    echo
    echo "${separator}"
    echo "${1}"
    echo "${separator}"
}

for arch in ${architectures}; do
    step "Building libc++.dylib for architecture ${arch}"
    mkdir -p "${build_dir}/${arch}"
    (cd "${build_dir}/${arch}" &&
        xcrun --sdk "${sdk}" cmake "${llvm_root}/libcxx" \
            -GNinja \
            -DCMAKE_MAKE_PROGRAM="$(xcrun --sdk "${sdk}" --find ninja)" \
            -C "${cache}" \
            -DCMAKE_INSTALL_PREFIX="${build_dir}/${arch}-install" \
            -DCMAKE_INSTALL_NAME_DIR="${install_name_dir}" \
            -DCMAKE_OSX_ARCHITECTURES="${arch}" \
            -DLIBCXX_INCLUDE_BENCHMARKS=OFF \
            -DLIBCXX_INCLUDE_TESTS=OFF
    )

    xcrun --sdk "${sdk}" cmake --build "${build_dir}/${arch}" --target install-cxx -- -v
done

step "Creating a universal dylib from the dylibs for all architectures"
input_dylibs=$(for arch in ${architectures}; do
    echo "${build_dir}/${arch}-install/lib/${dylib_name}"
done)
xcrun --sdk "${sdk}" lipo -create ${input_dylibs} -output "${build_dir}/${dylib_name}"

step "Installing the (stripped) universal dylib to ${install_dir}/usr/lib"
mkdir -p "${install_dir}/usr/lib"
cp "${build_dir}/${dylib_name}" "${install_dir}/usr/lib/${dylib_name}"
xcrun --sdk "${sdk}" strip -S "${install_dir}/usr/lib/${dylib_name}"
if [[ "${make_symlink}" == "yes" ]]; then
    (cd "${install_dir}/usr/lib" && ln -s "${dylib_name}" libc++.dylib)
fi

step "Installing the unstripped dylib and the dSYM bundle to ${symbols_dir}"
xcrun --sdk "${sdk}" dsymutil "${build_dir}/${dylib_name}" -o "${symbols_dir}/${dylib_name}.dSYM"
cp "${build_dir}/${dylib_name}" "${symbols_dir}/${dylib_name}"

#
# Install the headers by copying the headers from one of the built architectures
# into the install directory. Headers from all architectures should be the same.
#
any_arch=$(echo ${architectures} | cut -d ' ' -f 1)
mkdir -p "${install_dir}/usr/include"
ditto "${build_dir}/${any_arch}-install/include" "${install_dir}/usr/include"
