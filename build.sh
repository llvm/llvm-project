#!/usr/bin/env bash
#===-- build.sh ----------------------------------------------------------===#
# LLVM project build helper — debug/release × x86/aarch64/aarch64_be × minimal.
#
# Usage:
#   ./build.sh <type> <arch> [variant] [-c|-b]
#
#   <type>     debug | release
#   <arch>     x86 | aarch64 | aarch64_be
#   [variant]  (release only) default | minimal   (default: default)
#
#   -c         configure only (skip build)
#   -b         build only (skip configure)
#   -h         show help
#
# Examples:
#   ./build.sh debug x86                        # debug x86 → build/
#   ./build.sh debug x86 -c                     # configure only
#   ./build.sh debug x86 -b                     # build only
#   ./build.sh release x86                      # release x86 → build_x86/
#   ./build.sh release x86 minimal              # release x86 最小化 → build_x86_minimal/
#   ./build.sh release aarch64                  # release aarch64 → build_aarch64/
#   ./build.sh release aarch64 minimal          # release aarch64 最小化
#   ./build.sh release aarch64_be               # release aarch64 大端
#   ./build.sh release aarch64_be minimal       # release aarch64 大端最小化
#
# Output dirs:
#   debug  x86           → build/
#   release x86           → build_x86/
#   release x86 minimal   → build_x86_minimal/
#   release aarch64       → build_aarch64/
#   release aarch64 minimal → build_aarch64_minimal/
#   release aarch64_be    → build_aarch64_be/
#   release aarch64_be minimal → build_aarch64_be_minimal/
#===----------------------------------------------------------------------===#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'
log()  { echo -e "${GREEN}[build]${NC} $*"; }
warn() { echo -e "${YELLOW}[build]${NC} $*"; }
err()  { echo -e "${RED}[build]${NC} $*"; }

#===-- CMake helpers ------------------------------------------------------===#

LLVM_SRC="${ROOT_DIR}/llvm"

# Returns the CMake target triple for a given arch.
target_triple() {
  case "$1" in
    x86)         echo "x86_64-unknown-linux-gnu" ;;
    aarch64)     echo "aarch64-linux-gnu" ;;
    aarch64_be)  echo "aarch64_be-linux-gnu" ;;
  esac
}

# Returns the LLVM backend name for a given arch.
llvm_target() {
  case "$1" in
    x86)         echo "X86" ;;
    aarch64|aarch64_be) echo "AArch64" ;;
  esac
}

# Returns cross-compile flags for a given arch (empty for native x86).
cross_flags() {
  local arch="$1"
  if [ "$arch" = "x86" ]; then
    echo ""
  else
    local triple; triple=$(target_triple "$arch")
    echo "-DCMAKE_CROSSCOMPILING=ON"
    echo "-DLLVM_HOST_TRIPLE=${triple}"
    echo "-DLLVM_DEFAULT_TARGET_TRIPLE=${triple}"
    echo "-DEJIT_DEFAULT_TARGET_TRIPLE=${triple}"
  fi
}

# Cross-compiler toolchain prefix.
cross_cc() {
  local arch="$1"
  case "$arch" in
    x86)
      echo "clang clang++" ;;
    aarch64)
      echo "aarch64-linux-gnu-gcc aarch64-linux-gnu-g++" ;;
    aarch64_be)
      echo "aarch64_be-linux-gnu-gcc aarch64_be-linux-gnu-g++" ;;
  esac
}

# Big-endian flags.
endian_flags() {
  case "$1" in
    aarch64_be) echo "-mbig-endian" ;;
    *)          echo "" ;;
  esac
}

#===-- Build functions ----------------------------------------------------===#

do_configure_debug() {
  local arch="$1" build_dir="$2"
  local target; target=$(llvm_target "$arch")

  log "Configuring: debug ${arch} → ${build_dir}"

  cmake -S "${LLVM_SRC}" -B "${build_dir}" \
    -G "Ninja" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    "-DLLVM_TARGETS_TO_BUILD=${target}" \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    $(cross_flags "$arch")
}

do_build_debug() {
  local build_dir="$1"
  log "Building debug: ${build_dir}..."
  ninja -C "${build_dir}" clang opt
}

do_configure_release() {
  local arch="$1" build_dir="$2" variant="$3"
  local target; target=$(llvm_target "$arch")
  local cc cxx; read -r cc cxx <<< "$(cross_cc "$arch")"
  local beflag; beflag=$(endian_flags "$arch")

  local extra_flags=""
  local build_config="Release"

  if [ "$variant" = "minimal" ]; then
    build_config="Release"
    extra_flags="
      -DLLVM_ENABLE_ASSERTIONS=OFF
      -DLLVM_INCLUDE_TESTS=OFF
      -DLLVM_INCLUDE_EXAMPLES=OFF
      -DLLVM_INCLUDE_DOCS=OFF
      -DLLVM_INCLUDE_BENCHMARKS=OFF
      -DLLVM_ENABLE_PROJECTS=\"\"
      -DLLVM_ENABLE_ZLIB=OFF
      -DLLVM_ENABLE_ZSTD=OFF
      -DLLVM_ENABLE_LIBXML2=OFF
      -DLLVM_ENABLE_TERMINFO=OFF
      -DCMAKE_C_FLAGS=\"-ffunction-sections -fdata-sections ${beflag}\"
      -DCMAKE_CXX_FLAGS=\"-ffunction-sections -fdata-sections ${beflag}\"
      -DCMAKE_C_FLAGS_RELEASE=\"-Os -DNDEBUG\"
      -DCMAKE_CXX_FLAGS_RELEASE=\"-Os -DNDEBUG\"
      -DCMAKE_EXE_LINKER_FLAGS=\"-Wl,--gc-sections -Wl,--strip-all\"
      -DCMAKE_SHARED_LINKER_FLAGS=\"-Wl,--gc-sections -Wl,--strip-all\""
    log "Configuring: release ${arch} minimal → ${build_dir}"
  else
    extra_flags="
      -DCMAKE_C_FLAGS=\"${beflag}\"
      -DCMAKE_CXX_FLAGS=\"${beflag}\""
    log "Configuring: release ${arch} → ${build_dir}"
  fi

  # shellcheck disable=SC2086
  cmake -S "${LLVM_SRC}" -B "${build_dir}" \
    -G "Ninja" \
    -DCMAKE_BUILD_TYPE="${build_config}" \
    "-DLLVM_TARGETS_TO_BUILD=${target}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    "-DCMAKE_C_COMPILER=${cc}" \
    "-DCMAKE_CXX_COMPILER=${cxx}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    $(cross_flags "$arch") \
    ${extra_flags}
}

do_build_release() {
  local build_dir="$1" variant="$2"
  log "Building release: ${build_dir}..."
  ninja -C "${build_dir}" LLVMEJIT
  if [ "$variant" = "minimal" ]; then
    ninja -C "${build_dir}" ejit_minimal
  fi
}

#===-- Help ---------------------------------------------------------------===#

usage() {
  cat <<'EOF'
Usage: ./build.sh <type> <arch> [variant] [-c|-b]

  <type>     debug | release
  <arch>     x86 | aarch64 | aarch64_be
  [variant]  (release only) default | minimal

Options:
  -c   configure only (skip build)
  -b   build only (skip configure)
  -h   show this help

Examples:
  ./build.sh debug x86                  # debug x86 → build/
  ./build.sh debug x86 -c               # configure only
  ./build.sh debug x86 -b               # build only
  ./build.sh release x86                # release x86 → build_x86/
  ./build.sh release x86 minimal        # release x86 最小化 → build_x86_minimal/
  ./build.sh release aarch64            # release aarch64 → build_aarch64/
  ./build.sh release aarch64 minimal    # release aarch64 最小化
  ./build.sh release aarch64_be         # release aarch64 大端
  ./build.sh release aarch64_be minimal # release aarch64 大端 + 最小化

Output directories:
  debug  x86                         → build/
  release x86                        → build_x86/
  release x86 minimal                → build_x86_minimal/
  release aarch64                    → build_aarch64/
  release aarch64 minimal            → build_aarch64_minimal/
  release aarch64_be                 → build_aarch64_be/
  release aarch64_be minimal         → build_aarch64_be_minimal/
EOF
  exit 0
}

#===-- Main ---------------------------------------------------------------===#

TYPE="${1:-}"
ARCH="${2:-}"
VARIANT="default"
DO_CONFIGURE=true
DO_BUILD=true

shift 2 2>/dev/null || true

# Parse remaining args: [variant] [-c|-b]
while [[ $# -gt 0 ]]; do
  case "$1" in
    minimal) VARIANT="minimal" ;;
    default) VARIANT="default" ;;
    -c) DO_BUILD=false ;;
    -b) DO_CONFIGURE=false ;;
    -h|--help) usage ;;
    *) err "Unknown argument: $1"; usage ;;
  esac
  shift
done

# Validate
if [ -z "$TYPE" ] || [ -z "$ARCH" ]; then
  err "Missing arguments."
  usage
fi

case "$TYPE" in
  debug|release) ;;
  *) err "Invalid type: $TYPE (must be debug or release)"; usage ;;
esac

case "$ARCH" in
  x86|aarch64|aarch64_be) ;;
  *) err "Invalid arch: $ARCH (must be x86, aarch64, or aarch64_be)"; usage ;;
esac

if [ "$TYPE" = "debug" ] && [ "$VARIANT" = "minimal" ]; then
  warn "debug + minimal is not a typical combination; proceeding anyway."
fi

# Determine build directory
BIN_DIR() {
  case "${TYPE}_${ARCH}_${VARIANT}" in
    debug_x86_default|debug_x86_)              echo "${ROOT_DIR}/build" ;;
    release_x86_default|release_x86_)           echo "${ROOT_DIR}/build_x86" ;;
    release_x86_minimal)                        echo "${ROOT_DIR}/build_x86_minimal" ;;
    release_aarch64_default|release_aarch64_)   echo "${ROOT_DIR}/build_aarch64" ;;
    release_aarch64_minimal)                    echo "${ROOT_DIR}/build_aarch64_minimal" ;;
    release_aarch64_be_default|release_aarch64_be_) echo "${ROOT_DIR}/build_aarch64_be" ;;
    release_aarch64_be_minimal)                 echo "${ROOT_DIR}/build_aarch64_be_minimal" ;;
    debug_aarch64_*)                            echo "${ROOT_DIR}/build_aarch64" ;;
    debug_aarch64_be_*)                         echo "${ROOT_DIR}/build_aarch64_be" ;;
    *) echo "${ROOT_DIR}/build_${TYPE}_${ARCH}_${VARIANT}" ;;
  esac
}

BUILD_DIR=$(BIN_DIR)

log "Type=${TYPE}  Arch=${ARCH}  Variant=${VARIANT}"
log "Build dir: ${BUILD_DIR}"

if $DO_CONFIGURE; then
  case "$TYPE" in
    debug)   do_configure_debug "$ARCH"   "$BUILD_DIR" ;;
    release) do_configure_release "$ARCH" "$BUILD_DIR" "$VARIANT" ;;
  esac
fi

if $DO_BUILD; then
  if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
    err "No build.ninja found in ${BUILD_DIR}. Run with -c first (or omit -b)."
    exit 1
  fi
  case "$TYPE" in
    debug)   do_build_debug   "$BUILD_DIR" ;;
    release) do_build_release "$BUILD_DIR" "$VARIANT" ;;
  esac
fi

log "Done."
