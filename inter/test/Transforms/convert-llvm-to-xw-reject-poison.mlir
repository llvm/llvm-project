// RUN: not inter-opt %s --inter-import-llvm --inter-convert-llvm-to-xw 2>&1 | FileCheck %s

module {
  llvm.func spir_kernelcc @bad() {
    %value = llvm.mlir.poison : i32
    llvm.return
  }
}

// CHECK: undef and poison have no sound XW representation
