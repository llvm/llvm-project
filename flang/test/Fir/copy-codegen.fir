// Test fir.copy codegen.
// RUN: fir-opt --fir-to-llvm-ir %s -o - | FileCheck %s

!t=!fir.type<sometype{i:!fir.array<9xi32>}>
!t_tail_pad = !fir.type<tail_pad,sequence{a:i32,b:i8}>

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {

func.func @test_copy_1(%arg0: !fir.ref<!t>, %arg1: !fir.ref<!t>) {
  fir.copy %arg0 to %arg1 no_overlap : !fir.ref<!t>, !fir.ref<!t>
  return
}
// CHECK-LABEL:   llvm.func @test_copy_1(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(36 : i64) : i64
// CHECK:           "llvm.intr.memcpy"(%[[VAL_1]], %[[VAL_0]], %[[VAL_2]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           llvm.return
// CHECK:         }

func.func @test_copy_2(%arg0: !fir.ref<!t>, %arg1: !fir.ref<!t>) {
  fir.copy %arg0 to %arg1 : !fir.ref<!t>, !fir.ref<!t>
  return
}
// CHECK-LABEL:   llvm.func @test_copy_2(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(36 : i64) : i64
// CHECK:           "llvm.intr.memmove"(%[[VAL_1]], %[[VAL_0]], %[[VAL_2]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           llvm.return
// CHECK:         }

// SEQUENCE derived type with tail padding: i32 + i8 = 5 bytes of data,
// aligned to 4 bytes => 8-byte allocated size.  fir.copy must use the
// full allocated size so that the 3 padding bytes are included.
func.func @test_copy_tail_pad(%arg0: !fir.ref<!t_tail_pad>, %arg1: !fir.ref<!t_tail_pad>) {
  fir.copy %arg0 to %arg1 no_overlap : !fir.ref<!t_tail_pad>, !fir.ref<!t_tail_pad>
  return
}
// CHECK-LABEL:   llvm.func @test_copy_tail_pad(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK:           "llvm.intr.memcpy"(%[[VAL_1]], %[[VAL_0]], %[[VAL_2]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           llvm.return
// CHECK:         }
}
