// RUN: fir-opt --split-input-file --cfg-conversion %s | FileCheck %s

func.func private @callee() -> none

// CHECK-LABEL:   func.func @if_then(
// CHECK-SAME:      %[[ARG0:.*]]: i1) {
// CHECK:           cf.cond_br %[[ARG0]] weights([10, 90]), ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_0:.*]] = fir.call @callee() : () -> none
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           return
// CHECK:         }
func.func @if_then(%cond: i1) {
  fir.if %cond weights([10, 90]) {
    fir.call @callee() : () -> none
  }
  return
}

// -----

// CHECK-LABEL:   func.func @if_then_else(
// CHECK-SAME:      %[[ARG0:.*]]: i1) -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           cf.cond_br %[[ARG0]] weights([90, 10]), ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3(%[[VAL_0]] : i32)
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3(%[[VAL_1]] : i32)
// CHECK:         ^bb3(%[[VAL_2:.*]]: i32):
// CHECK:           cf.br ^bb4
// CHECK:         ^bb4:
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }
func.func @if_then_else(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %result = fir.if %cond weights([90, 10]) -> i32 {
    fir.result %c0 : i32
  } else {
    fir.result %c1 : i32
  }
  return %result : i32
}
