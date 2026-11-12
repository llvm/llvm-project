// This test only makes sure that passes with a DLTI dialect dependency are able
// to obtain the dlti.dl_spec module attribute from an llvm.data_layout string.
//
// If dependencies for the pass are not properly set, this test causes a
// compiler error due to the DLTI dialect not being loaded.

// RUN: fir-opt --add-debug-info %s
// RUN: fir-opt --cuf-add-constructor %s
// RUN: fir-opt --cuf-compute-shared-memory %s
// RUN: fir-opt --cuf-gpu-convert-to-llvm %s
// RUN: fir-opt --cuf-convert %s
// RUN: fir-opt --loop-versioning %s

module attributes {llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"} {
  llvm.func @foo(%arg0 : i32) {
    llvm.return
  }
}

// CHECK: module attributes {
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec<
