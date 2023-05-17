// Test hlfir.assign rewrite in the bufferization pass.
// Assign in itself is not transformed, but its operands may be
// expressions that are bufferized and must be updated.
// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @keep_attributes(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,8>>>>>, %arg1: !fir.box<!fir.array<1x!fir.char<1,?>>>) {
  %true = arith.constant true
  %0 = hlfir.as_expr %arg1 move %true : (!fir.box<!fir.array<1x!fir.char<1,?>>>, i1) -> !hlfir.expr<1x!fir.char<1,?>>
  hlfir.assign %0 to %arg0 realloc keep_lhs_len : !hlfir.expr<1x!fir.char<1,?>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,8>>>>>
  return
}

// CHECK-LABEL: func.func @keep_attributes(
// CHECK-SAME: %[[X:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,8>>>>>,
// CHECK-SAME: %[[Y:.*]]: !fir.box<!fir.array<1x!fir.char<1,?>>>) {
// CHECK: hlfir.assign %[[Y]] to %[[X]] realloc keep_lhs_len : !fir.box<!fir.array<1x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,8>>>>>
