// RUN: fir-opt --split-input-file --cuf-add-constructor="extra-constructors=foo entry-only-constructors=bar" %s | FileCheck %s

// extra-constructors is a list of <name>:<if-main> pairs. The functions are
// declared and called from __cudaFortranConstructor, since they are defined
// in a runtime library. if-main=false always calls the function; if-main=true
// calls it only when the module contains the Fortran program entry (_QQmain).

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"} {
}

// CHECK: llvm.func internal @__cudaFortranConstructor() {
// CHECK:   llvm.call @foo() : () -> ()
// CHECK-NOT: llvm.call @bar()
// CHECK:   llvm.return
// CHECK: }
// CHECK: llvm.func @foo()
// CHECK: llvm.mlir.global_ctors ctors = [@__cudaFortranConstructor], priorities = [0 : i32], data = [#llvm.zero]

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"} {
  func.func @_QQmain() {
    return
  }
}

// CHECK: llvm.func internal @__cudaFortranConstructor() {
// CHECK:   llvm.call @foo() : () -> ()
// CHECK:   llvm.call @bar() : () -> ()
// CHECK:   llvm.return
// CHECK: }
// CHECK-DAG: llvm.func @foo()
// CHECK-DAG: llvm.func @bar()
// CHECK: llvm.mlir.global_ctors ctors = [@__cudaFortranConstructor], priorities = [0 : i32], data = [#llvm.zero]

// -----

// An existing declaration is reused rather than declared a second time.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"} {
  llvm.func @foo()
}

// CHECK: llvm.func @foo()
// CHECK-NOT: llvm.func @foo()
// CHECK: llvm.func internal @__cudaFortranConstructor() {
// CHECK:   llvm.call @foo() : () -> ()
// CHECK:   llvm.return
// CHECK: }
