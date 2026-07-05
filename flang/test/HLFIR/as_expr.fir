// Test hlfir.as_expr operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @char_expr(%arg0: !fir.boxchar<1>) {
  %0 = hlfir.as_expr %arg0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL: func.func @char_expr(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>) {
// CHECK:   hlfir.as_expr %[[VAL_0]] : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>

func.func @char_expr_2(%arg0: !fir.ref<!fir.char<1,10>>) {
  %0 = hlfir.as_expr %arg0 : (!fir.ref<!fir.char<1,10>>) -> !hlfir.expr<!fir.char<1,10>>
  return
}
// CHECK-LABEL: func.func @char_expr_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,10>>) {
// CHECK:   hlfir.as_expr %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !hlfir.expr<!fir.char<1,10>>

func.func @array_expr(%arg0: !fir.box<!fir.array<?xi32>>) {
  %0 = hlfir.as_expr %arg0 : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @array_expr(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:   hlfir.as_expr %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>

func.func @array_expr_2(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %0 = hlfir.as_expr %arg0 : (!fir.ref<!fir.array<10xi32>>) -> !hlfir.expr<10xi32>
  return
}
// CHECK-LABEL: func.func @array_expr_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) {
// CHECK:   hlfir.as_expr %[[VAL_0]] : (!fir.ref<!fir.array<10xi32>>) -> !hlfir.expr<10xi32>

func.func @array_expr_move(%arg0: !fir.ref<!fir.array<10xi32>>, %must_free: i1) {
  %0 = hlfir.as_expr %arg0 move %must_free : (!fir.ref<!fir.array<10xi32>>, i1) -> !hlfir.expr<10xi32>
  return
}
// CHECK-LABEL: func.func @array_expr_move(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i1) {
// CHECK:   hlfir.as_expr %[[VAL_0]] move %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, i1) -> !hlfir.expr<10xi32>
