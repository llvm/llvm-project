// Test hlfir.product operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

// mask is an expression of known shape
func.func @any0(%arg0: !hlfir.expr<2x!fir.logical<4>>) {
  %any = hlfir.any %arg0 : (!hlfir.expr<2x!fir.logical<4>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @any0(%[[ARRAY:.*]]: !hlfir.expr<2x!fir.logical<4>>) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] : (!hlfir.expr<2x!fir.logical<4>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is an expression of assumed shape
func.func @any1(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  %any = hlfir.any %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @any1(%[[ARRAY:.*]]: !hlfir.expr<?x!fir.logical<4>>) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a boxed array
func.func @any2(%arg0: !fir.box<!fir.array<2x!fir.logical<4>>>) {
  %any = hlfir.any %arg0 : (!fir.box<!fir.array<2x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @any2(%[[ARRAY:.*]]: !fir.box<!fir.array<2x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] : (!fir.box<!fir.array<2x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is an assumed shape boxed array
func.func @any3(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>){
  %any = hlfir.any %arg0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @any3(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a 2-dimensional array
func.func @any4(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>>){
  %any = hlfir.any %arg0 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @any4(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask and dim argument
func.func @any5(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>, %arg1: i32) {
  %any = hlfir.any %arg0 dim %arg1 : (!fir.box<!fir.array<?x!fir.logical<4>>>, i32) -> !fir.logical<4>
  return
}
// CHECK:      func.func @any5(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, i32) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.any with dim argument with an unusual type
func.func @any6(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>, %arg1: index) {
  %any = hlfir.any %arg0 dim %arg1 : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) ->!fir.logical<4>
  return
}
// CHECK:      func.func @any6(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>, %[[DIM:.*]]: index) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a 2 dimensional array with dim
func.func @any7(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>>, %arg1: i32) {
  %any = hlfir.any %arg0 dim %arg1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK:      func.func @any7(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// known shape expr return
func.func @any8(%arg0: !fir.box<!fir.array<2x2x!fir.logical<4>>>, %arg1: i32) {
  %any = hlfir.any %arg0 dim %arg1 : (!fir.box<!fir.array<2x2x!fir.logical<4>>>, i32) -> !hlfir.expr<2x!fir.logical<4>>
  return
}
// CHECK:      func.func @any8(%[[ARRAY:.*]]: !fir.box<!fir.array<2x2x!fir.logical<4>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<2x2x!fir.logical<4>>>, i32) -> !hlfir.expr<2x!fir.logical<4>>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.any with mask argument of ref<array<>> type
func.func @any9(%arg0: !fir.ref<!fir.array<?x!fir.logical<4>>>) {
  %any = hlfir.any %arg0 : (!fir.ref<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK:      func.func @any9(%[[ARRAY:.*]]: !fir.ref<!fir.array<?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] : (!fir.ref<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.any with fir.logical<8> type
func.func @any10(%arg0: !fir.box<!fir.array<?x!fir.logical<8>>>) {
  %any = hlfir.any %arg0 : (!fir.box<!fir.array<?x!fir.logical<8>>>) -> !fir.logical<8>
  return
}
// CHECK:      func.func @any10(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<8>>>) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.any %[[ARRAY]] : (!fir.box<!fir.array<?x!fir.logical<8>>>) -> !fir.logical<8>
// CHECK-NEXT:   return
// CHECK-NEXT: }
