#!/usr/bin/env bash
#===-- build.sh ----------------------------------------------------------===#
# LLVM + EmbeddedJIT build helper.
#
# Usage:
#   ./build.sh release aarch64         # → build_release_aarch64/  (native, recommended)
#   ./build.sh release aarch64_be      # → build_release_aarch64_be/ (SRE code pool ON by default)
#   ./build.sh release x86             # → build_release_x86/
#   ./build.sh debug   x86             # → build_debug_x86/
#   ./build.sh debug   x86 --static    # → build_debug_x86_static/
#   ./build.sh release x86 minimal     # → build_release_x86_minimal/
#   ./build.sh debug   aarch64         # → build_debug_aarch64/
#   ./build.sh release aarch64 minimal # → build_release_aarch64_minimal/
#
# Options:
#   -c              configure only (skip build)
#   -b              build only (skip configure)
#   --static        debug build with static libs (for ejit_test with assertions)
#   --no-ccache     disable ccache
#   --trim-llvm-backend-experimental    build with EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL=ON (backend: trim non-ELF formats)
#   --freestanding  build with EJIT_FREESTANDING=ON (runtime: no OS threads/I/O)
#   --target-triple=<triple>  set EJIT_DEFAULT_TARGET_TRIPLE
#   --sre-code-pool / --no-sre-code-pool  force EmbeddedJIT SRE code pool on/off
#   --sre-code-pool-ptno=<n>  set SRE code pool partition number (default: 8)
#   --sre-taskpool / --no-sre-taskpool  enable/disable SRE taskpool scheduler (default OFF)
#   --sre-taskpool-buckets=<n>  taskpool dedup/cache bucket count (default: 32)
#   --sre-taskpool-bucket-slots=<n>  taskpool slots per bucket (default: 8)
#   --sre-taskpool-queue-capacity=<n>  taskpool async queue capacity, pow2 (default: 1024)
#   -h              show help
#===----------------------------------------------------------------------===#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLVM_SRC="${ROOT_DIR}/llvm"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'
log()  { echo -e "${GREEN}[build]${NC} $*"; }
warn() { echo -e "${YELLOW}[build]${NC} $*"; }
err()  { echo -e "${RED}[build]${NC} $*"; }

#===-- Helpers ------------------------------------------------------------===#

target_triple() {
  case "$1" in
    x86)     echo "x86_64-unknown-linux-gnu" ;;
    aarch64) echo "aarch64-linux-gnu" ;;
    aarch64_be) echo "aarch64_be-unknown-linux-gnu" ;;
  esac
}

llvm_target() {
  case "$1" in
    x86)     echo "X86" ;;
    aarch64) echo "AArch64" ;;
    aarch64_be) echo "AArch64" ;;
  esac
}

# Default compiler for native builds.  Override via CMAKE_C_COMPILER / CMAKE_CXX_COMPILER
# environment variables for cross-compilation (e.g. aarch64-linux-gnu-gcc).
native_cc() {
  echo "clang clang++"
}

# Build directory naming convention.
build_dir() {
  local type="$1" arch="$2" variant="${3:-default}"
  if [ "$variant" = "minimal" ]; then
    echo "${ROOT_DIR}/build_${type}_${arch}_minimal"
  elif [ "$variant" = "static" ]; then
    echo "${ROOT_DIR}/build_${type}_${arch}_static"
  else
    echo "${ROOT_DIR}/build_${type}_${arch}"
  fi
}

#===-- Configure -----------------------------------------------------------===#

do_configure() {
  local type="$1" arch="$2" build_dir="$3" variant="${4:-default}"
  local target; target=$(llvm_target "$arch")
  local cc cxx; read -r cc cxx <<< "$(native_cc)"
  local default_triple; default_triple=$(target_triple "$arch")

  local ccache_opts=""
  if $USE_CCACHE; then
    ccache_opts="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
  fi

  if [ "$type" = "debug" ]; then
    if [ "$variant" = "static" ]; then
      log "Configuring: debug ${arch} static → ${build_dir}"
      cmake -S "${LLVM_SRC}" -B "${build_dir}" \
        -G "Ninja" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        "-DLLVM_TARGETS_TO_BUILD=${target}" \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DEJIT_SRE_CODE_POOL=${EJIT_SRE_CODE_POOL} \
        -DEJIT_SRE_CODE_POOL_PTNO=${EJIT_SRE_CODE_POOL_PTNO} \
        -DEJIT_SRE_TASKPOOL=${EJIT_SRE_TASKPOOL} \
        -DEJIT_SRE_TASKPOOL_BUCKETS=${EJIT_SRE_TASKPOOL_BUCKETS} \
        -DEJIT_SRE_TASKPOOL_BUCKET_SLOTS=${EJIT_SRE_TASKPOOL_BUCKET_SLOTS} \
        -DEJIT_SRE_TASKPOOL_QUEUE_CAPACITY=${EJIT_SRE_TASKPOOL_QUEUE_CAPACITY} \
        "-DEJIT_DEFAULT_TARGET_TRIPLE=${EJIT_TARGET_TRIPLE:-${default_triple}}" \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        ${ccache_opts}
    else
      log "Configuring: debug ${arch} → ${build_dir}"
      cmake -S "${LLVM_SRC}" -B "${build_dir}" \
        -G "Ninja" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_SHARED_LIBS=ON \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        "-DLLVM_TARGETS_TO_BUILD=${target}" \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DEJIT_SRE_CODE_POOL=${EJIT_SRE_CODE_POOL} \
        -DEJIT_SRE_CODE_POOL_PTNO=${EJIT_SRE_CODE_POOL_PTNO} \
        -DEJIT_SRE_TASKPOOL=${EJIT_SRE_TASKPOOL} \
        -DEJIT_SRE_TASKPOOL_BUCKETS=${EJIT_SRE_TASKPOOL_BUCKETS} \
        -DEJIT_SRE_TASKPOOL_BUCKET_SLOTS=${EJIT_SRE_TASKPOOL_BUCKET_SLOTS} \
        -DEJIT_SRE_TASKPOOL_QUEUE_CAPACITY=${EJIT_SRE_TASKPOOL_QUEUE_CAPACITY} \
        "-DEJIT_DEFAULT_TARGET_TRIPLE=${EJIT_TARGET_TRIPLE:-${default_triple}}" \
        -DLLVM_USE_SPLIT_DWARF=ON \
        -DLLVM_ENABLE_ZLIB=OFF \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        ${ccache_opts}
    fi
    return
  fi

  # release
  # extra_flags: flags without spaces in values (safe for unquoted expansion).
  # Flags with spaces (CFLAGS, CXXFLAGS, LINKER_FLAGS) go directly in the
  # cmake command line below with proper shell quoting.
  #
  # We use two separate cmake invocations (minimal / normal) because the
  # minimal variant has extra linker flags whose values contain spaces.
  local extra_flags=""

  if [ "$variant" = "minimal" ]; then
    log "Configuring: release ${arch} minimal → ${build_dir}"
    extra_flags="
      -DLLVM_ENABLE_PROJECTS=
      -DLLVM_INCLUDE_TESTS=OFF
      -DLLVM_INCLUDE_EXAMPLES=OFF
      -DLLVM_INCLUDE_DOCS=OFF
      -DLLVM_INCLUDE_BENCHMARKS=OFF
      -DLLVM_ENABLE_ZLIB=OFF
      -DLLVM_ENABLE_ZSTD=OFF
      -DLLVM_ENABLE_LIBXML2=OFF
      -DLLVM_ENABLE_TERMINFO=OFF
      -DEJIT_SRE_TASKPOOL=${EJIT_SRE_TASKPOOL}
      -DEJIT_SRE_TASKPOOL_BUCKETS=${EJIT_SRE_TASKPOOL_BUCKETS}
      -DEJIT_SRE_TASKPOOL_BUCKET_SLOTS=${EJIT_SRE_TASKPOOL_BUCKET_SLOTS}
      -DEJIT_SRE_TASKPOOL_QUEUE_CAPACITY=${EJIT_SRE_TASKPOOL_QUEUE_CAPACITY}
    "
    # shellcheck disable=SC2086
    cmake -S "${LLVM_SRC}" -B "${build_dir}" \
      -G "Ninja" \
      -DCMAKE_BUILD_TYPE=Release \
      "-DLLVM_TARGETS_TO_BUILD=${target}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLVM_OPTIMIZED_TABLEGEN=ON \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      "-DCMAKE_C_COMPILER=${cc}" \
      "-DCMAKE_CXX_COMPILER=${cxx}" \
      -DEJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL=${EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL} \
      -DEJIT_FREESTANDING=${EJIT_FREESTANDING} \
      -DEJIT_SRE_CODE_POOL=${EJIT_SRE_CODE_POOL} \
      -DEJIT_SRE_CODE_POOL_PTNO=${EJIT_SRE_CODE_POOL_PTNO} \
      "-DEJIT_DEFAULT_TARGET_TRIPLE=${EJIT_TARGET_TRIPLE:-${default_triple}}" \
      ${ccache_opts} \
      ${extra_flags} \
      "-DCMAKE_C_FLAGS=-ffunction-sections -fdata-sections" \
      "-DCMAKE_CXX_FLAGS=-ffunction-sections -fdata-sections" \
      "-DCMAKE_C_FLAGS_RELEASE=-Os -DNDEBUG" \
      "-DCMAKE_CXX_FLAGS_RELEASE=-Os -DNDEBUG" \
      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--gc-sections -Wl,--strip-all" \
      "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--gc-sections -Wl,--strip-all"
  else
    log "Configuring: release ${arch} → ${build_dir}"
    extra_flags+="
      -DLLVM_ENABLE_ZSTD=OFF
      -DLLVM_ENABLE_ZLIB=OFF
      -DLLVM_ENABLE_TERMINFO=OFF
      -DLLVM_ENABLE_LIBXML2=OFF
      -DLLVM_ENABLE_PIC=OFF
      -DLLVM_ENABLE_THREADS=OFF
      -DEJIT_SRE_TASKPOOL=${EJIT_SRE_TASKPOOL}
      -DEJIT_SRE_TASKPOOL_BUCKETS=${EJIT_SRE_TASKPOOL_BUCKETS}
      -DEJIT_SRE_TASKPOOL_BUCKET_SLOTS=${EJIT_SRE_TASKPOOL_BUCKET_SLOTS}
      -DEJIT_SRE_TASKPOOL_QUEUE_CAPACITY=${EJIT_SRE_TASKPOOL_QUEUE_CAPACITY}
    "
    # shellcheck disable=SC2086
    cmake -S "${LLVM_SRC}" -B "${build_dir}" \
      -G "Ninja" \
      -DCMAKE_BUILD_TYPE=Release \
      "-DLLVM_TARGETS_TO_BUILD=${target}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLVM_OPTIMIZED_TABLEGEN=ON \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      "-DCMAKE_C_COMPILER=${cc}" \
      "-DCMAKE_CXX_COMPILER=${cxx}" \
      -DEJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL=${EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL} \
      -DEJIT_FREESTANDING=${EJIT_FREESTANDING} \
      -DEJIT_SRE_CODE_POOL=${EJIT_SRE_CODE_POOL} \
      -DEJIT_SRE_CODE_POOL_PTNO=${EJIT_SRE_CODE_POOL_PTNO} \
      "-DEJIT_DEFAULT_TARGET_TRIPLE=${EJIT_TARGET_TRIPLE:-${default_triple}}" \
      ${ccache_opts} \
      ${extra_flags} \
      "-DCMAKE_C_FLAGS=-ffunction-sections -fdata-sections" \
      "-DCMAKE_CXX_FLAGS=-ffunction-sections -fdata-sections" \
      "-DCMAKE_C_FLAGS_RELEASE=-Os -DNDEBUG" \
      "-DCMAKE_CXX_FLAGS_RELEASE=-Os -DNDEBUG"
  fi
}

#===-- Build ---------------------------------------------------------------===#

do_build() {
  local type="$1" build_dir="$2" variant="${3:-default}"

  case "$type" in
    debug)
      if [ "$variant" = "static" ]; then
        log "Building debug static: ${build_dir}..."
        ninja -C "${build_dir}" clang LLVMEJIT lld
      else
        log "Building debug: ${build_dir}..."
        ninja -C "${build_dir}" clang opt lld
      fi
      ;;
    release)
      if [ "$variant" = "minimal" ]; then
        log "Building release minimal: ${build_dir}..."
        ninja -C "${build_dir}" ejit_minimal
      else
        log "Building release: ${build_dir}..."
        ninja -C "${build_dir}" clang LLVMEJIT lld
      fi
      ;;
  esac
}

#===-- Main ---------------------------------------------------------------===#

TYPE="${1:-}"
ARCH="${2:-}"
VARIANT="default"
DO_CONFIGURE=true
DO_BUILD=true
USE_CCACHE=true
EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL=OFF
EJIT_FREESTANDING=OFF
EJIT_TARGET_TRIPLE=""
EJIT_SRE_CODE_POOL=AUTO
EJIT_SRE_CODE_POOL_PTNO=8
EJIT_SRE_TASKPOOL=OFF
EJIT_SRE_TASKPOOL_BUCKETS=32
EJIT_SRE_TASKPOOL_BUCKET_SLOTS=8
EJIT_SRE_TASKPOOL_QUEUE_CAPACITY=1024

if [[ "${1:-}" = "-h" || "${1:-}" = "--help" ]]; then
  sed -n '2,29p' "$0"
  exit 0
fi

shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    minimal) VARIANT="minimal" ;;
    --static) VARIANT="static" ;;
    -c) DO_BUILD=false ;;
    -b) DO_CONFIGURE=false ;;
    --no-ccache) USE_CCACHE=false ;;
    --trim-llvm-backend-experimental) EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL=ON ;;
    --freestanding) EJIT_FREESTANDING=ON ;;
    --target-triple=*) EJIT_TARGET_TRIPLE="${1#--target-triple=}" ;;
    --sre-code-pool) EJIT_SRE_CODE_POOL=ON ;;
    --no-sre-code-pool) EJIT_SRE_CODE_POOL=OFF ;;
    --sre-code-pool-ptno=*) EJIT_SRE_CODE_POOL_PTNO="${1#--sre-code-pool-ptno=}" ;;
    --sre-taskpool) EJIT_SRE_TASKPOOL=ON ;;
    --no-sre-taskpool) EJIT_SRE_TASKPOOL=OFF ;;
    --sre-taskpool-buckets=*) EJIT_SRE_TASKPOOL_BUCKETS="${1#--sre-taskpool-buckets=}" ;;
    --sre-taskpool-bucket-slots=*) EJIT_SRE_TASKPOOL_BUCKET_SLOTS="${1#--sre-taskpool-bucket-slots=}" ;;
    --sre-taskpool-queue-capacity=*) EJIT_SRE_TASKPOOL_QUEUE_CAPACITY="${1#--sre-taskpool-queue-capacity=}" ;;
    -h|--help)
      sed -n '2,29p' "$0"
      exit 0
      ;;
    *) err "Unknown argument: $1"; exit 1 ;;
  esac
  shift
done

if [ -z "$TYPE" ] || [ -z "$ARCH" ]; then
  err "Usage: ./build.sh <debug|release> <x86|aarch64> [minimal] [-c|-b]"
  exit 1
fi

case "$TYPE" in  debug|release) ;;  *) err "Invalid type: $TYPE"; exit 1 ;; esac
case "$ARCH" in  x86|aarch64|aarch64_be) ;;    *) err "Invalid arch: $ARCH"; exit 1 ;; esac

if [ "$EJIT_SRE_CODE_POOL" = "AUTO" ]; then
  if [ "$ARCH" = "aarch64_be" ]; then
    EJIT_SRE_CODE_POOL=ON
  else
    EJIT_SRE_CODE_POOL=OFF
  fi
fi

if [ "$TYPE" = "debug" ] && [ "$VARIANT" = "minimal" ]; then
  warn "debug + minimal is unusual; proceeding anyway."
fi

BUILD_DIR=$(build_dir "$TYPE" "$ARCH" "$VARIANT")
log "Type=${TYPE}  Arch=${ARCH}  Variant=${VARIANT}  ccache=$($USE_CCACHE && echo on || echo off)"
log "EJIT: triple=${EJIT_TARGET_TRIPLE:-$(target_triple "$ARCH")}  sre-code-pool=${EJIT_SRE_CODE_POOL}  ptno=${EJIT_SRE_CODE_POOL_PTNO}"
log "EJIT: sre-taskpool=${EJIT_SRE_TASKPOOL} buckets=${EJIT_SRE_TASKPOOL_BUCKETS} slots=${EJIT_SRE_TASKPOOL_BUCKET_SLOTS} queue=${EJIT_SRE_TASKPOOL_QUEUE_CAPACITY}"
log "Build dir: ${BUILD_DIR}"

if $DO_CONFIGURE; then
  do_configure "$TYPE" "$ARCH" "$BUILD_DIR" "$VARIANT"
fi

if $DO_BUILD; then
  if [ ! -f "${BUILD_DIR}/build.ninja" ]; then
    err "No build.ninja in ${BUILD_DIR}. Run with -c first (or omit -b)."
    exit 1
  fi
  do_build "$TYPE" "$BUILD_DIR" "$VARIANT"
fi

log "Done."
