// RUN: fir-opt --split-input-file --cuf-add-constructor="extra-constructors=foo:false,bar:true" %s | FileCheck %s

// extra-constructors is a list of <name>:<if-main> pairs. if-main=false
// always appends the function to llvm.mlir.global_ctors; if-main=true
// appends it only when the module contains the Fortran program entry
// (_QQmain). Each extra ctor reuses the same priority and #llvm.zero
// data as the CUDA Fortran constructor.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"} {
  llvm.func @foo() {
    llvm.return
  }
  llvm.func @bar() {
    llvm.return
  }
}

// CHECK: llvm.func internal @__cudaFortranConstructor()
// CHECK: llvm.mlir.global_ctors ctors = [@__cudaFortranConstructor, @foo], priorities = [0 : i32, 0 : i32], data = [#llvm.zero, #llvm.zero]

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"} {
  func.func @_QQmain() {
    return
  }
  llvm.func @foo() {
    llvm.return
  }
  llvm.func @bar() {
    llvm.return
  }
}

// CHECK: llvm.func internal @__cudaFortranConstructor()
// CHECK: llvm.mlir.global_ctors ctors = [@__cudaFortranConstructor, @foo, @bar], priorities = [0 : i32, 0 : i32, 0 : i32], data = [#llvm.zero, #llvm.zero, #llvm.zero]
