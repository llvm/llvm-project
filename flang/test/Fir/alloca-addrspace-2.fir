// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s
// RUN: tco --fir-to-llvm-ir %s | FileCheck %s

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>> } {
  // CHECK-LABEL: llvm.func @set_addrspace
  func.func @set_addrspace() {
    // CHECK: llvm.alloca {{.*}} x i32
    // CHECK-SAME: -> !llvm.ptr<5>
    %0 = fir.alloca i32
    return
  }
}
