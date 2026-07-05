// Test hlfir.no_reassoc operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @no_reassoc_value(%arg0: i32) {
  %0 = hlfir.no_reassoc %arg0 : i32
  return
}
// CHECK-LABEL: func.func @no_reassoc_value(
// CHECK-SAME:    %[[VAL_0:.*]]: i32) {
// CHECK:   hlfir.no_reassoc %[[VAL_0]] : i32

func.func @no_reassoc_var(%arg0: !fir.ref<i32>) {
  %0 = hlfir.no_reassoc %arg0 : !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @no_reassoc_var(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>) {
// CHECK:   hlfir.no_reassoc %[[VAL_0]] : !fir.ref<i32>

func.func @no_reassoc_expr(%arg0: !hlfir.expr<10xi32>) {
  %0 = hlfir.no_reassoc %arg0 : !hlfir.expr<10xi32>
  return
}
// CHECK-LABEL: func.func @no_reassoc_expr(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<10xi32>) {
// CHECK:   hlfir.no_reassoc %[[VAL_0]] : !hlfir.expr<10xi32>
