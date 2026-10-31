#!/usr/bin/env bash
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Sourced by run-buildbot, which runs with `set -e`.
#
# Some self-hosted CI agents ship a CMake older than what runtimes/CMakeLists.txt
# requires. When that happens, provision a pinned, checksum-verified CMake into
# a per-user cache. When the host CMake is recent enough, this costs a single
# `cmake --version` and changes nothing else.
#
# Keep CMAKE_MIN_VERSION in sync with cmake_minimum_required() in
# runtimes/CMakeLists.txt.
CMAKE_MIN_VERSION=3.31
CMAKE_PINNED_VERSION=3.31.12

function cmake-is-supported() {
    local version
    version=$("$1" --version 2>/dev/null) || return 1
    [[ "$version" =~ ^cmake\ version\ ([0-9]+)\.([0-9]+)\. ]] || return 1
    (( BASH_REMATCH[1] > ${CMAKE_MIN_VERSION%%.*} ||
       (BASH_REMATCH[1] == ${CMAKE_MIN_VERSION%%.*} &&
        BASH_REMATCH[2] >= ${CMAKE_MIN_VERSION##*.}) ))
}

# python3 is already required by the test suite (lit), so it is used here for
# the pieces that have no portable shell equivalent across Linux, macOS, AIX,
# FreeBSD and Git Bash on Windows.
function download-cmake-archive() {
    local filename="$1" destination="$2" sha256="$3"
    # Fail on a stalled transfer instead of hanging until the CI timeout.
    curl --fail --location --retry 3 --connect-timeout 30 \
        --speed-limit 1024 --speed-time 60 \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_PINNED_VERSION}/${filename}" \
        --output "$destination"
    python3 -c 'import hashlib, sys
if hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != sys.argv[2]:
    sys.exit("CMake archive checksum mismatch: " + sys.argv[1])' "$destination" "$sha256"
}

# Runs in a subshell so the EXIT trap, cd, and CC/CXX/flag changes do not leak
# into the caller.
function install-cmake() (
    set -e
    local install_dir="$1" system="$2" arch="$3"
    local version="$CMAKE_PINNED_VERSION"
    local archive='' sha256=''
    mkdir -p "$(dirname "$install_dir")"
    # Not `local`: bash unsets locals before running the EXIT trap when the
    # function is invoked with a redirection. The subshell scopes it anyway.
    work=$(mktemp -d "${install_dir}.tmp.XXXXXX")
    trap 'rm -rf "$work"' EXIT

    # Discard an incomplete or unusable installation from an earlier run.
    if [[ -e "$install_dir" ]]; then
        mv "$install_dir" "${work}/stale"
    fi

    # Select binaries by host, not by the architecture being tested.
    case "${system}-${arch}" in
        Linux-x86_64)
            archive="cmake-${version}-linux-x86_64.tar.gz"
            sha256=0dc2e9a6860f06bf10bd8fadc03e35d9eeb4df46e33763a7e480e987758f385c
            ;;
        Linux-aarch64)
            archive="cmake-${version}-linux-aarch64.tar.gz"
            sha256=83f8fd91d2038a56556e1400390fcfe42f79602940c494f6c6f1cdae7f9e7f40
            ;;
        Darwin-x86_64|Darwin-arm64)
            archive="cmake-${version}-macos10.10-universal.tar.gz"
            sha256=cf096d869d45b79f5ce82f3d6abffa44f983fc146fee13cfe59024911e6975af
            ;;
        MINGW*-x86_64|MSYS*-x86_64|CYGWIN*-x86_64)
            archive="cmake-${version}-windows-x86_64.zip"
            sha256=0c4baa40f28b3f8225eb3fdf6946c987b4fe901403b4eaf2fbbd9378100aaa0c
            ;;
        MINGW*-aarch64|MSYS*-aarch64|CYGWIN*-aarch64)
            archive="cmake-${version}-windows-arm64.zip"
            sha256=e4160c1842dea858ad376ff2ec17587104515b51714eca5963b8bdd798105553
            ;;
    esac

    if [[ -n "$archive" ]]; then
        download-cmake-archive "$archive" "${work}/${archive}" "$sha256"
        if [[ "$archive" == *.zip ]]; then
            # Git for Windows does not ship unzip.
            python3 -m zipfile -e "${work}/${archive}" "${work}/binary"
            mv "${work}/binary/cmake-${version}-windows-"* "${work}/install"
        elif [[ "$system" == Darwin ]]; then
            mkdir "${work}/binary"
            tar -xzf "${work}/${archive}" -C "${work}/binary" --strip-components=1
            mv "${work}/binary/CMake.app/Contents" "${work}/install"
        else
            mkdir "${work}/install"
            tar -xzf "${work}/${archive}" -C "${work}/install" --strip-components=1
        fi
    fi

    # No binary for this host, or one that is incompatible with its libc/loader.
    if ! cmake-is-supported "${work}/install/bin/cmake"; then
        echo "Bootstrapping CMake ${version} from source for ${system}-${arch}"
        download-cmake-archive "cmake-${version}.tar.gz" "${work}/source.tar.gz" \
            5f3fd5a54dfa65602bdbed64f981a72673cc19f2d304cc2955cf0dfa0cfd8272
        rm -rf "${work}/install"
        mkdir -p "${work}/source" "${work}/build"
        tar -xzf "${work}/source.tar.gz" -C "${work}/source" --strip-components=1
        # CC/CXX and flags from run-buildbot may target a different machine.
        export CC="${HOST_CC:-cc}" CXX="${HOST_CXX:-c++}"
        unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CMAKE_TOOLCHAIN_FILE
        local jobs make_program=make
        jobs="${CMAKE_BOOTSTRAP_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
        if command -v gmake >/dev/null 2>&1; then
            make_program=gmake
        fi
        # The runtime builds don't need CMake's HTTPS support. Avoid requiring
        # OpenSSL development headers on the host just to bootstrap CMake.
        # Build in a nested subshell so the EXIT trap does not run from inside
        # the directory it removes.
        (
            cd "${work}/build"
            "${work}/source/bootstrap" --prefix="${work}/install" --parallel="$jobs" \
                -- -DCMAKE_USE_OPENSSL=OFF -DBUILD_TESTING=OFF
            "$make_program" -j "$jobs"
            "$make_program" install
        )
        cmake-is-supported "${work}/install/bin/cmake"
    fi

    # Publish the complete, relocatable install atomically. If another builder
    # populated the cache while we were working, the rename fails; in that case
    # discard our copy (via the EXIT trap) and use theirs.
    python3 -c 'import os, sys
try:
    os.rename(sys.argv[1], sys.argv[2])
except OSError as error:
    sys.exit("CMake install already present: " + str(error))' \
        "${work}/install" "$install_dir" \
        || cmake-is-supported "${install_dir}/bin/cmake"
)

function ensure-cmake() {
    local cmake_dir
    if [[ -n ${CMAKE+x} ]]; then
        if ! cmake-is-supported "$CMAKE"; then
            echo "CMAKE must name a runnable CMake ${CMAKE_MIN_VERSION} or newer" \
                 "(output must start with 'cmake version'): ${CMAKE}" >&2
            return 1
        fi
        # Scripts we invoke that run a bare `cmake` must see the same binary.
        cmake_dir=$(cd "$(dirname "$(command -v "$CMAKE")")" && pwd)
    elif cmake-is-supported cmake; then
        # Fast path: use the host's cmake as-is; do not reorder PATH.
        CMAKE=cmake
        export CMAKE
        return 0
    else
        local system arch install_dir
        system=$(uname -s)
        arch=$(uname -m)
        # Keep tools outside BUILD_DIR, which run-buildbot's clean() removes.
        install_dir="${XDG_CACHE_HOME:-${HOME:-${TMPDIR:-/tmp}}/.cache}/libcxx/cmake-${CMAKE_PINNED_VERSION}-${system}-${arch}"
        if ! cmake-is-supported "${install_dir}/bin/cmake"; then
            echo "cmake on PATH is missing or older than ${CMAKE_MIN_VERSION};" \
                 "provisioning CMake ${CMAKE_PINNED_VERSION} into ${install_dir}" >&2
            # Called as a plain statement: wrapping it in `||` or `if !` would
            # disable errexit inside the subshell.
            install-cmake "$install_dir" "$system" "$arch"
        fi
        CMAKE="${install_dir}/bin/cmake"
        cmake-is-supported "$CMAKE" || return 1
        cmake_dir="${install_dir}/bin"
    fi
    PATH="${cmake_dir}:${PATH}"
    export CMAKE PATH
}
