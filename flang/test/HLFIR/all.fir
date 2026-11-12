// Test hlfir.all operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

// mask is an expression of known shape
func.func @all0(%arg0: !hlfir.expr<2x!fir.logical<4>>) {
  %all = hlfir.all %arg0 : (!hlfir.expr<2x!fir.logical<4>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @all0(%[[ARRAY:.*]]: !hlfir.expr<2x!fir.logical<4>>) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] : (!hlfir.expr<2x!fir.logical<4>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is an expression of assumed shape
func.func @all1(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  %all = hlfir.all %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @all1(%[[ARRAY:.*]]: !hlfir.expr<?x!fir.logical<4>>) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a boxed array
func.func @all2(%arg0: !fir.box<!fir.array<2x!fir.logical<4>>>) {
  %all = hlfir.all %arg0 : (!fir.box<!fir.array<2x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @all2(%[[ARRAY:.*]]: !fir.box<!fir.array<2x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] : (!fir.box<!fir.array<2x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is an assumed shape boxed array
func.func @all3(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>){
  %all = hlfir.all %arg0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @all3(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a 2-dimensional array
func.func @all4(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>>){
  %all = hlfir.all %arg0 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @all4(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask and dim argument
func.func @all5(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>, %arg1: i32) {
  %all = hlfir.all %arg0 dim %arg1 : (!fir.box<!fir.array<?x!fir.logical<4>>>, i32) -> !fir.logical<4>
  return
}
// CHECK:      func.func @all5(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, i32) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.all with dim argument with an unusual type
func.func @all6(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>, %arg1: index) {
  %all = hlfir.all %arg0 dim %arg1 : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) ->!fir.logical<4>
  return
}
// CHECK:      func.func @all6(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>, %[[DIM:.*]]: index) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a 2 dimensional array with dim
func.func @all7(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>>, %arg1: i32) {
  %all = hlfir.all %arg0 dim %arg1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK:      func.func @all7(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// known shape expr return
func.func @all8(%arg0: !fir.box<!fir.array<2x2x!fir.logical<4>>>, %arg1: i32) {
  %all = hlfir.all %arg0 dim %arg1 : (!fir.box<!fir.array<2x2x!fir.logical<4>>>, i32) -> !hlfir.expr<2x!fir.logical<4>>
  return
}
// CHECK:      func.func @all8(%[[ARRAY:.*]]: !fir.box<!fir.array<2x2x!fir.logical<4>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<2x2x!fir.logical<4>>>, i32) -> !hlfir.expr<2x!fir.logical<4>>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.all with mask argument of ref<array<>> type
func.func @all9(%arg0: !fir.ref<!fir.array<?x!fir.logical<4>>>) {
  %all = hlfir.all %arg0 : (!fir.ref<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @all9(%[[ARRAY:.*]]: !fir.ref<!fir.array<?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] : (!fir.ref<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.all with fir.logical<8> type
func.func @all10(%arg0: !fir.box<!fir.array<?x!fir.logical<8>>>) {
  %all = hlfir.all %arg0 : (!fir.box<!fir.array<?x!fir.logical<8>>>) -> !fir.logical<8>
  return
}
// CHECK:      func.func @all10(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<8>>>) {
// CHECK-NEXT:   %[[ALL:.*]] = hlfir.all %[[ARRAY]] : (!fir.box<!fir.array<?x!fir.logical<8>>>) -> !fir.logical<8>
// CHECK-NEXT:   return
// CHECK-NEXT: }