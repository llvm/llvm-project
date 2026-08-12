// RUN: not inter-opt %s --inter-import-llvm --inter-convert-llvm-to-xw 2>&1 | FileCheck %s

module {
  llvm.func spir_kernelcc @bad(%lhs: f32, %rhs: f32) {
    %value = llvm.fdiv %lhs, %rhs : f32
    llvm.return
  }
}

// CHECK: floating division and remainder have no exact XW operation
