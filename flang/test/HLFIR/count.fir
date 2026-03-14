// Test hlfir.count operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

// mask is an expression of known shape
func.func @count0(%arg0: !hlfir.expr<2x!fir.logical<4>>) {
  %count = hlfir.count %arg0 : (!hlfir.expr<2x!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @count0(%[[ARRAY:.*]]: !hlfir.expr<2x!fir.logical<4>>) {
// CHECK-NEXT:   %[[COUNT:.*]] = hlfir.count %[[ARRAY]] : (!hlfir.expr<2x!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is an expression of assumed shape
func.func @count1(%arg0: !hlfir.expr<?x!fir.logical<4>>) {
  %count = hlfir.count %arg0 : (!hlfir.expr<?x!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @count1(%[[ARRAY:.*]]: !hlfir.expr<?x!fir.logical<4>>) {
// CHECK-NEXT:   %[[COUNT:.*]] = hlfir.count %[[ARRAY]] : (!hlfir.expr<?x!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a boxed array
func.func @count2(%arg0: !fir.box<!fir.array<2x!fir.logical<4>>>) {
  %count = hlfir.count %arg0 : (!fir.box<!fir.array<2x!fir.logical<4>>>) -> i32
  return
}
// CHECK:      func.func @count2(%[[ARRAY:.*]]: !fir.box<!fir.array<2x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[COUNT:.*]] = hlfir.count %[[ARRAY]] : (!fir.box<!fir.array<2x!fir.logical<4>>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is an assumed shape boxed array
func.func @count3(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %count = hlfir.count %arg0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
  return
}
// CHECK:      func.func @count3(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[COUNT:.*]] = hlfir.count %[[ARRAY]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a 2-dimensional array
func.func @count4(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>>){
  %count = hlfir.count %arg0 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> i32
  return
}
// CHECK:      func.func @count4(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[COUNT:.*]] = hlfir.count %[[ARRAY]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask and dim argument
func.func @count5(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>, %arg1: i32) {
  %count = hlfir.count %arg0 dim %arg1 : (!fir.box<!fir.array<?x!fir.logical<4>>>, i32) -> i32
  return
}
// CHECK:      func.func @count5(%[[ARRAY:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[COUNT:.*]] = hlfir.count %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, i32) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is a 2 dimensional array with dim
func.func @count6(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>>, %arg1: i32) {
  %count = hlfir.count %arg0 dim %arg1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> i32
  return
}
// CHECK:      func.func @count6(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.count %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// mask is of a different logical type
func.func @count7(%arg0: !fir.box<!fir.array<?x?x!fir.logical<8>>>, %arg1: i32) {
  %count = hlfir.count %arg0 dim %arg1 : (!fir.box<!fir.array<?x?x!fir.logical<8>>>, i32) -> i32
  return
}
// CHECK:      func.func @count7(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?x!fir.logical<8>>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[ANY:.*]] = hlfir.count %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x?x!fir.logical<8>>>, i32) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }