// RUN: not inter-opt %s --inter-import-llvm --inter-convert-llvm-to-xw 2>&1 | FileCheck %s

module {
  llvm.func spir_kernelcc @bad() {
    %value = llvm.mlir.undef : i32
    llvm.return
  }
}

// CHECK: undef has no sound XW representation
