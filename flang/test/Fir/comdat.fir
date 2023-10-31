
// RUN: fir-opt %s --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %s --check-prefixes="CHECK-COMDAT"
// RUN: fir-opt %s --fir-to-llvm-ir="target=x86_64-pc-windows-msvc" | FileCheck %s --check-prefixes="CHECK-COMDAT"
// RUN: fir-opt %s --fir-to-llvm-ir="target=aarch64-apple-darwin" | FileCheck %s --check-prefixes="CHECK-NOCOMDAT"
// RUN: fir-opt %s --fir-to-llvm-ir="target=powerpc64-ibm-aix" | FileCheck %s --check-prefixes="CHECK-NOCOMDAT"

// CHECK-COMDAT: llvm.func linkonce @fun_linkonce(%arg0: i32) -> i32 comdat(@__llvm_comdat::@fun_linkonce)
// CHECK-NOCOMDAT: llvm.func linkonce @fun_linkonce(%arg0: i32) -> i32 {
func.func @fun_linkonce(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce>} {
  return %arg0 : i32
}

// CHECK-COMDAT: llvm.func linkonce_odr @fun_linkonce_odr(%arg0: i32) -> i32 comdat(@__llvm_comdat::@fun_linkonce_odr)
// CHECK-NOCOMDAT: llvm.func linkonce_odr @fun_linkonce_odr(%arg0: i32) -> i32 {
func.func @fun_linkonce_odr(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
  return %arg0 : i32
}

// CHECK-COMDAT:  llvm.mlir.global linkonce constant @global_linkonce() comdat(@__llvm_comdat::@global_linkonce) {addr_space = 0 : i32} : i32 
// CHECK-NOCOMDAT:  llvm.mlir.global linkonce constant @global_linkonce() {addr_space = 0 : i32} : i32  {
fir.global linkonce @global_linkonce constant : i32 {
  %0 = arith.constant 0 : i32
  fir.has_value %0 : i32
}

// CHECK-COMDAT: llvm.comdat @__llvm_comdat {
// CHECK-COMDAT: llvm.comdat_selector @fun_linkonce_odr any
// CHECK-COMDAT: llvm.comdat_selector @fun_linkonce any
// CHECK-COMDAT: llvm.comdat_selector @global_linkonce any
// CHECK-COMDAT: llvm.comdat_selector @global_linkonce_odr any

// CHECK-NOCOMDAT-NOT: llvm.comdat
// CHECK-NOCOMDAT-NOT: llvm.comdat_selector


// CHECK-COMDAT:  llvm.mlir.global linkonce_odr constant @global_linkonce_odr() comdat(@__llvm_comdat::@global_linkonce_odr) {addr_space = 0 : i32} : i32 
// CHECK-NOCOMDAT:  llvm.mlir.global linkonce_odr constant @global_linkonce_odr() {addr_space = 0 : i32} : i32 { 
fir.global linkonce_odr @global_linkonce_odr constant : i32 {
  %0 = arith.constant 0 : i32
  fir.has_value %0 : i32
}