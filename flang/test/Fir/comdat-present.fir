// RUN: fir-opt %s --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %s
// RUN: fir-opt %s --fir-to-llvm-ir="target=x86_64-pc-windows-msvc" | FileCheck %s

fir.global linkonce_odr @global_linkonce_odr constant : i32 {
  %0 = arith.constant 0 : i32
  fir.has_value %0 : i32
}

llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @global_linkonce_odr any
}

// CHECK-LABEL: llvm.comdat @__llvm_comdat
// CHECK: llvm.comdat_selector @global_linkonce_odr any
