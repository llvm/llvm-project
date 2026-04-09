// Test codegen of constant insert_on_range without symbol reference into mlir.constant.
// RUN: fir-opt --cg-rewrite --split-input-file --fir-to-llvm-ir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  fir.global @derived_array : !fir.array<2x!fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>> {
    %c0 = arith.constant 0 : index
    %0 = fir.undefined !fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>
    %1 = fir.zero_bits !fir.heap<!fir.array<?xf64>>
    %2 = fir.shape %c0 : (index) -> !fir.shape<1>
    %3 = fir.embox %1(%2) : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf64>>>
    %4 = fir.insert_value %0, %3, ["comp", !fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>] : (!fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>, !fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>
    %5 = fir.undefined !fir.array<2x!fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>
    %6 = fir.insert_on_range %5, %4 from (0) to (1) : (!fir.array<2x!fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>, !fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>) -> !fir.array<2x!fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>
    fir.has_value %6 : !fir.array<2x!fir.type<sometype{comp:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>
  }
}

//CHECK-LABEL: llvm.mlir.global external @derived_array()
//CHECK:    %[[CST:.*]] = llvm.mlir.constant([
//CHECK-SAME:  [
//CHECK-SAME:    [#llvm.zero, 8, 20240719 : i32, 1 : i8, 28 : i8, 2 : i8, 0 : i8,
//CHECK-SAME:      [
//CHECK-SAME:        [1, 0 : index, 8]
//CHECK-SAME:      ]
//CHECK-SAME:    ],
//CHECK-SAME:  [
//CHECK-SAME:    [#llvm.zero, 8, 20240719 : i32, 1 : i8, 28 : i8, 2 : i8, 0 : i8,
//CHECK-SAME:      [
//CHECK-SAME:        [1, 0 : index, 8]
//CHECK-SAME:      ]
//CHECK-SAME:    ]) :
//CHECK-SAME: !llvm.array<2 x struct<"sometype", (struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>)>>
//CHECK:    llvm.return %[[CST]] : !llvm.array<2 x struct<"sometype", (struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>)>>
