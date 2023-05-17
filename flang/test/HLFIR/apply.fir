// Test hlfir.apply operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @numeric(%expr: !hlfir.expr<?x?xf32>) {
  %c9 = arith.constant 9 : index
  %c2 = arith.constant 2 : index
  %0 = hlfir.apply %expr, %c9, %c2 : (!hlfir.expr<?x?xf32>, index, index) -> f32
  return
}
// CHECK-LABEL:   func.func @numeric(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<?x?xf32>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 9 : index
// CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_3:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : (!hlfir.expr<?x?xf32>, index, index) -> f32

func.func @char(%expr: !hlfir.expr<?x?x!fir.char<1,?>>, %l: index) {
  %c9 = arith.constant 9 : index
  %c2 = arith.constant 2 : index
  %0 = hlfir.apply %expr, %c9, %c2 typeparams %l: (!hlfir.expr<?x?x!fir.char<1,?>>, index, index, index) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @char(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<?x?x!fir.char<1,?>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_4:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_2]], %[[VAL_3]] typeparams %[[VAL_1]] : (!hlfir.expr<?x?x!fir.char<1,?>>, index, index, index) -> !hlfir.expr<!fir.char<1,?>>

!pdt = !fir.type<pdt(param:i32){field:f32}>
func.func @derived(%expr: !hlfir.expr<?x?x!pdt>, %l: i32) {
  %c9 = arith.constant 9 : index
  %c2 = arith.constant 2 : index
  %0 = hlfir.apply %expr, %c9, %c2 typeparams %l: (!hlfir.expr<?x?x!pdt>, index, index, i32) -> !hlfir.expr<!pdt>
  return
}
// CHECK-LABEL:   func.func @derived(
// CHECK-SAME:    %[[VAL_0:.*]]: !hlfir.expr<?x?x!fir.type<pdt(param:i32){field:f32}>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i32) {
// CHECK:  %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:  %[[VAL_4:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_2]], %[[VAL_3]] typeparams %[[VAL_1]] : (!hlfir.expr<?x?x!fir.type<pdt(param:i32){field:f32}>>, index, index, i32) -> !hlfir.expr<!fir.type<pdt(param:i32){field:f32}>>
