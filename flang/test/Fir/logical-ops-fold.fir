// RUN: fir-opt --canonicalize %s | FileCheck %s

// logical_and(false, false) == false
// CHECK-LABEL: @fold_logical_and_false_false
func.func @fold_logical_and_false_false() -> !fir.logical<4> {
  %false = arith.constant false
  %lhs = fir.convert %false : (i1) -> !fir.logical<4>
  %rhs = fir.convert %false : (i1) -> !fir.logical<4>
  %res = fir.logical_and %lhs, %rhs : !fir.logical<4>
  // CHECK-NOT: fir.logical_and
  // CHECK: %[[FALSE:.*]] = fir.convert
  // CHECK: return %[[FALSE]] : !fir.logical<4>
  return %res : !fir.logical<4>
}

// logical_and(true, false) == false
// CHECK-LABEL: @fold_logical_and_true_false
func.func @fold_logical_and_true_false() -> !fir.logical<4> {
  %true = arith.constant true
  %false = arith.constant false
  %lhs = fir.convert %true : (i1) -> !fir.logical<4>
  %rhs = fir.convert %false : (i1) -> !fir.logical<4>
  %res = fir.logical_and %lhs, %rhs : !fir.logical<4>
  // CHECK-NOT: fir.logical_and
  // CHECK: %[[FALSE:.*]] = arith.constant false
  // CHECK: %[[RHS:.*]] = fir.convert %[[FALSE]] : (i1) -> !fir.logical<4>
  // CHECK: return %[[RHS]] : !fir.logical<4>
  return %res : !fir.logical<4>
}

// logical_and(x, true) -> x
// CHECK-LABEL: @fold_logical_and_identity
// CHECK-SAME: (%[[X:.*]]: !fir.logical<4>)
func.func @fold_logical_and_identity(%x: !fir.logical<4>) -> !fir.logical<4> {
  %true = arith.constant true
  %rhs = fir.convert %true : (i1) -> !fir.logical<4>
  %res = fir.logical_and %x, %rhs : !fir.logical<4>
  // CHECK-NOT: fir.logical_and
  // CHECK: return %[[X]] : !fir.logical<4>
  return %res : !fir.logical<4>
}

// logical_or(true, false) == true
// CHECK-LABEL: @fold_logical_or_true_false
func.func @fold_logical_or_true_false() -> !fir.logical<4> {
  %true = arith.constant true
  %false = arith.constant false
  %lhs = fir.convert %true : (i1) -> !fir.logical<4>
  %rhs = fir.convert %false : (i1) -> !fir.logical<4>
  %res = fir.logical_or %lhs, %rhs : !fir.logical<4>
  // CHECK-NOT: fir.logical_or
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[LHS:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
  // CHECK: return %[[LHS]] : !fir.logical<4>
  return %res : !fir.logical<4>
}

// logical_or(x, false) -> x
// CHECK-LABEL: @fold_logical_or_identity
// CHECK-SAME: (%[[X:.*]]: !fir.logical<4>)
func.func @fold_logical_or_identity(%x: !fir.logical<4>) -> !fir.logical<4> {
  %false = arith.constant false
  %rhs = fir.convert %false : (i1) -> !fir.logical<4>
  %res = fir.logical_or %x, %rhs : !fir.logical<4>
  // CHECK-NOT: fir.logical_or
  // CHECK: return %[[X]] : !fir.logical<4>
  return %res : !fir.logical<4>
}

// Fold on integer type: logical_and(1, 0) -> arith.constant 0 : i32
// CHECK-LABEL: @fold_logical_and_int
func.func @fold_logical_and_int() -> i32 {
  %c1 = arith.constant 1 : i1
  %c0 = arith.constant 0 : i1
  %lhs = fir.convert %c1 : (i1) -> i32
  %rhs = fir.convert %c0 : (i1) -> i32
  %res = fir.logical_and %lhs, %rhs : i32
  // CHECK-NOT: fir.logical_and
  // CHECK: %[[C:.*]] = arith.constant 0 : i32
  // CHECK: return %[[C]] : i32
  return %res : i32
}
