// RUN: not inter-opt %s '--inter-import-llvm=simd-width=64' 2>&1 | FileCheck %s

module {
  llvm.func spir_kernelcc @bad() {
    llvm.return
  }
}

// CHECK: --simd-width must be 8, 16, or 32
