#!/usr/bin/env bash
set -euo pipefail

# Build only static libraries of LLVM and Clang and install CMake packages

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${ROOT_DIR}/llvm"
BUILD_DIR="${ROOT_DIR}/build-static"
INSTALL_DIR="${ROOT_DIR}/install"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Error: expected to find LLVM source at ${SRC_DIR}. Run this from the llvm-project root." >&2
  exit 1
fi

# Configurables via env vars
BUILD_TYPE="${BUILD_TYPE:-Release}"
TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-host}"

if [[ "${CLEAN:-0}" == "1" ]]; then
  echo "[clean] Removing ${BUILD_DIR} and ${INSTALL_DIR}"
  rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
fi

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

GENERATOR="Ninja"
if ! command -v ninja >/dev/null 2>&1; then
  GENERATOR="Unix Makefiles"
fi

echo "[configure] Generator: ${GENERATOR}"
echo "[configure] Build type: ${BUILD_TYPE}"
echo "[configure] Targets: ${TARGETS_TO_BUILD}"
echo "[configure] Build dir: ${BUILD_DIR}"
echo "[configure] Install dir: ${INSTALL_DIR}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G "${GENERATOR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_TARGETS_TO_BUILD="${TARGETS_TO_BUILD}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_ENABLE_PIC=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DLLVM_BUILD_LLVM_C_DYLIB=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DCLANG_INCLUDE_TESTS=OFF \
  -DCLANG_INCLUDE_DOCS=OFF \
  -DCLANG_BUILD_TOOLS=OFF \
  -DLIBCLANG_BUILD_STATIC=ON \
  ${CMAKE_EXTRA_ARGS:-}

echo "[build] Building and installing to ${INSTALL_DIR}"
cmake --build "${BUILD_DIR}" --target install -j "$(nproc)"

echo "[done] Static LLVM/Clang libraries and CMake packages installed under: ${INSTALL_DIR}"


