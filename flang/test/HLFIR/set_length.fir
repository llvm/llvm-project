// Test hlfir.set_length operation parse, verify (no errors), and unparse.
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test_cst_len(%str : !fir.boxchar<1>) {
  %c10 = arith.constant 10 : index
  %0 = hlfir.set_length %str len %c10 : (!fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,10>>
  return
}
// CHECK-LABEL:   func.func @test_cst_len(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_2:.*]] = hlfir.set_length %[[VAL_0]] len %[[VAL_1]] : (!fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,10>>

func.func @test_dyn_len(%str : !fir.ref<!fir.char<1,10>>, %len : index) {
  %0 = hlfir.set_length %str len %len : (!fir.ref<!fir.char<1,10>>, index) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @test_dyn_len(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_2:.*]] = hlfir.set_length %[[VAL_0]] len %[[VAL_1]] : (!fir.ref<!fir.char<1,10>>, index) -> !hlfir.expr<!fir.char<1,?>>

func.func @test_from_expr(%str : !hlfir.expr<!fir.char<1,?>>, %len : index) {
  %0 = hlfir.set_length %str len %len : (!hlfir.expr<!fir.char<1,?>>, index) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @test_from_expr(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<!fir.char<1,?>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_2:.*]] = hlfir.set_length %[[VAL_0]] len %[[VAL_1]] : (!hlfir.expr<!fir.char<1,?>>, index) -> !hlfir.expr<!fir.char<1,?>>
