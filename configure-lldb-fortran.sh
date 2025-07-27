#!/bin/bash
# Optimal build configuration for LLDB Fortran development
# This script creates a minimal build with only necessary components

BUILD_DIR="${1:-build-fortran}"
BUILD_TYPE="${2:-RelWithDebInfo}"

echo "Configuring LLVM/LLDB for Fortran development..."
echo "Build directory: $BUILD_DIR"
echo "Build type: $BUILD_TYPE"

cmake -S llvm -B "$BUILD_DIR" -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;lldb;flang" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLDB_INCLUDE_TESTS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_PARALLEL_LINK_JOBS=2 \
  -DLLVM_PARALLEL_COMPILE_JOBS=6 \
  -DLLDB_TEST_FORTRAN=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Configuration complete. To build:"
echo "  ninja -C $BUILD_DIR lldb lldb-test"
echo "To run Fortran tests:"
echo "  ninja -C $BUILD_DIR check-lldb-lang-fortran"