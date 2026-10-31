// RUN: fir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ftest
func.func @ftest(%x : i1) -> i1 {
  // this pair of converts should be folded and DCEd
  %1 = fir.convert %x : (i1) -> !fir.logical<1>
  %2 = fir.convert %1 : (!fir.logical<1>) -> i1
  // CHECK-NEXT: return %{{.*}} : i1
  return %2 : i1
}

// CHECK-LABEL: @gtest
func.func @gtest(%x : !fir.logical<2>) -> !fir.logical<2> {
  // this pair of converts should be folded and DCEd
  %1 = fir.convert %x : (!fir.logical<2>) -> i1
  %2 = fir.convert %1 : (i1) -> !fir.logical<2>
  // CHECK-NEXT: return %{{.*}} : !fir.logical<2>
  return %2 : !fir.logical<2>
}

// CHECK-LABEL: @htest
func.func @htest(%x : !fir.int<4>) -> !fir.int<4> {
  // these converts are NOPs and should be folded away
  %1 = fir.convert %x : (!fir.int<4>) -> !fir.int<4>
  %2 = fir.convert %1 : (!fir.int<4>) -> !fir.int<4>
  // CHECK-NEXT: return %{{.*}} : !fir.int<4>
  return %2 : !fir.int<4>
}

// CHECK-LABEL: @ctest
func.func @ctest() -> index {
  %1 = arith.constant 10 : i32
  %2 = fir.convert %1 : (i32) -> index
  // CHECK-NEXT: %{{.*}} = arith.constant 10 : index
  // CHECK-NEXT: return %{{.*}} : index
  return %2 : index
}

// CHECK-LABEL:   func.func @ptrtest(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.heap<!fir.array<2xf32>>) -> !fir.heap<!fir.array<2xf32>> {
func.func @ptrtest(%0 : !fir.heap<!fir.array<2xf32>>) -> !fir.heap<!fir.array<2xf32>> {
  %1 = fir.convert %0 : (!fir.heap<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<2xf32>>) -> !fir.heap<!fir.array<2xf32>>
// CHECK:           return %[[VAL_0]] : !fir.heap<!fir.array<2xf32>>
  return %2 : !fir.heap<!fir.array<2xf32>>
}

// CHECK-LABEL: @convert_fold_widen_signed
func.func @convert_fold_widen_signed() -> i64 {
  %1 = arith.constant -1 : i32
  %2 = fir.convert %1 : (i32) -> i64
  // CHECK-NEXT: %{{.*}} = arith.constant -1 : i64
  // CHECK-NEXT: return %{{.*}} : i64
  return %2 : i64
}

// CHECK-LABEL: @convert_fold_widen_i1
func.func @convert_fold_widen_i1() -> i32 {
  %1 = arith.constant true
  %2 = fir.convert %1 : (i1) -> i32
  // CHECK-NEXT: %{{.*}} = arith.constant 1 : i32
  // CHECK-NEXT: return %{{.*}} : i32
  return %2 : i32
}

// CHECK-LABEL: @convert_fold_narrow
func.func @convert_fold_narrow() -> i8 {
  %1 = arith.constant 258 : i32
  %2 = fir.convert %1 : (i32) -> i8
  // CHECK-NEXT: %{{.*}} = arith.constant 2 : i8
  // CHECK-NEXT: return %{{.*}} : i8
  return %2 : i8
}

// The fold composes with arith.index_cast, which already folds a constant, so
// a value widened and then cast to index becomes a single index constant.
// CHECK-LABEL: @convert_fold_chain_to_index
func.func @convert_fold_chain_to_index() -> index {
  %1 = arith.constant 1 : i32
  %2 = fir.convert %1 : (i32) -> i64
  %3 = arith.index_cast %2 : i64 to index
  // CHECK-NEXT: %{{.*}} = arith.constant 1 : index
  // CHECK-NOT: fir.convert
  // CHECK: return %{{.*}} : index
  return %3 : index
}

// CHECK-LABEL: @convert_no_fold
func.func @convert_no_fold(%x : i32) -> i64 {
  %1 = fir.convert %x : (i32) -> i64
  // CHECK-NEXT: %[[R:.*]] = fir.convert %{{.*}} : (i32) -> i64
  // CHECK-NEXT: return %[[R]] : i64
  return %1 : i64
}

// An unsigned result is left alone: arith.constant cannot carry it.
// CHECK-LABEL: @convert_no_fold_unsigned_result
func.func @convert_no_fold_unsigned_result() -> ui32 {
  %1 = arith.constant 1 : i32
  %2 = fir.convert %1 : (i32) -> ui32
  // CHECK: %[[R:.*]] = fir.convert %{{.*}} : (i32) -> ui32
  // CHECK: return %[[R]] : ui32
  return %2 : ui32
}
