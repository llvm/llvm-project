#!/usr/bin/env bash
# Build the EJIT C demo and report binary size.
# Usage: ./build_demo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${ROOT_DIR}/build_x86"        # release static libs
CLANG="${ROOT_DIR}/build/bin/clang"      # debug clang (for EJIT attrs)
DEMO_SRC="${SCRIPT_DIR}/ejit_demo.c"
DEMO_BIN="${SCRIPT_DIR}/ejit_demo"

LLVM_INCLUDE_DIR="${ROOT_DIR}/llvm/include"
BUILD_INCLUDE_DIR="${BUILD_DIR}/include"
LLVM_LIB_DIR="${BUILD_DIR}/lib"

echo "=== Building EJIT Demo ==="

# Build the combined archive if not already done
EJIT_A="${BUILD_DIR}/lib/ExecutionEngine/EJIT/libejit_minimal.a"
if [ ! -f "${EJIT_A}" ]; then
  echo "Combining archives into libejit_minimal.a..."
  cd "${BUILD_DIR}" && ninja ejit_minimal
fi

# Collect all static libraries
LIBS=""
for lib in "${LLVM_LIB_DIR}"/*.a; do
  LIBS="${LIBS} ${lib}"
done

echo "Compiling demo..."
"${CLANG}" -Os \
  -I"${LLVM_INCLUDE_DIR}" \
  -I"${BUILD_INCLUDE_DIR}" \
  -DNDEBUG \
  -ffunction-sections -fdata-sections \
  -c "${DEMO_SRC}" -o "${SCRIPT_DIR}/ejit_demo.o"

echo "Linking demo (--whole-archive + --gc-sections)..."
clang++ -Os \
  -ffunction-sections -fdata-sections \
  -Wl,--gc-sections \
  -Wl,--strip-all \
  "${SCRIPT_DIR}/ejit_demo.o" \
  -Wl,--whole-archive ${LIBS} -Wl,--no-whole-archive \
  -lpthread -ldl \
  -o "${DEMO_BIN}"

echo ""
echo "=== Binary Size ==="
ls -lh "${DEMO_BIN}"
echo ""
echo "=== Section Sizes ==="
size -A "${DEMO_BIN}" | head -30
echo ""
echo "=== Stripped binary total ==="
wc -c < "${DEMO_BIN}" | xargs -I{} echo "  {} bytes"
