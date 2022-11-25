// Test hlfir.concat operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @concat_var(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1,20>>) {
  %c30 = arith.constant 30 : index
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, index) -> (!hlfir.expr<!fir.char<1,30>>)
  return
}
// CHECK-LABEL:   func.func @concat_var(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.char<1,20>>) {
// CHECK:  %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:  %[[VAL_3:.*]] = hlfir.concat %[[VAL_0]], %[[VAL_1]] len %[[VAL_2]] : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, index) -> !hlfir.expr<!fir.char<1,30>>


func.func @concat_boxchar(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>) {
  %c30 = arith.constant 30 : index
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> (!hlfir.expr<!fir.char<1,?>>)
  return
}
// CHECK-LABEL:   func.func @concat_boxchar(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:  %[[VAL_3:.*]] = hlfir.concat %[[VAL_0]], %[[VAL_1]] len %[[VAL_2]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>


func.func @concat_boxchar_kind2(%arg0: !fir.boxchar<2>, %arg1: !fir.boxchar<2>) {
  %c30 = arith.constant 30 : index
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.boxchar<2>, !fir.boxchar<2>, index) -> (!hlfir.expr<!fir.char<2,?>>)
  return
}
// CHECK-LABEL:   func.func @concat_boxchar_kind2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<2>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.boxchar<2>) {
// CHECK:  %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:  %[[VAL_3:.*]] = hlfir.concat %[[VAL_0]], %[[VAL_1]] len %[[VAL_2]] : (!fir.boxchar<2>, !fir.boxchar<2>, index) -> !hlfir.expr<!fir.char<2,?>>


func.func @concat_expr(%arg0: !hlfir.expr<!fir.char<1,10>>, %arg1: !hlfir.expr<!fir.char<1,20>>) {
  %c30 = arith.constant 30 : index
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>, index) -> (!hlfir.expr<!fir.char<1,30>>)
  return
}
// CHECK-LABEL:   func.func @concat_expr(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !hlfir.expr<!fir.char<1,20>>) {
// CHECK:  %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:  %[[VAL_3:.*]] = hlfir.concat %[[VAL_0]], %[[VAL_1]] len %[[VAL_2]] : (!hlfir.expr<!fir.char<1,10>>, !hlfir.expr<!fir.char<1,20>>, index) -> !hlfir.expr<!fir.char<1,30>>


func.func @concat_several_args(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>) {
  %c30 = arith.constant 30 : index
  %0 = hlfir.concat %arg0, %arg1, %arg1 len %c30 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>, index) -> (!hlfir.expr<!fir.char<1,?>>)
  return
}
// CHECK-LABEL:   func.func @concat_several_args(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_2:.*]] = arith.constant 30 : index
// CHECK:  %[[VAL_3:.*]] = hlfir.concat %[[VAL_0]], %[[VAL_1]], %[[VAL_1]] len %[[VAL_2]] : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
