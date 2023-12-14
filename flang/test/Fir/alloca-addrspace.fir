// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s
// RUN: tco --fir-to-llvm-ir %s | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @default_addrspace
  func.func @default_addrspace() {
    // CHECK: llvm.alloca {{.*}} x i32
    // CHECK-SAME: -> !llvm.ptr
    %0 = fir.alloca i32
    return
  }
}
