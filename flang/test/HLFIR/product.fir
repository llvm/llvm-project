// Test hlfir.product operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

// array is an expression of known shape
func.func @product0(%arg0: !hlfir.expr<42xi32>) {
  %mask = fir.alloca !fir.logical<4>
  %c_1 = arith.constant 1 : index
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.product %arg0 dim %c_1 mask %mask_box : (!hlfir.expr<42xi32>, index, !fir.box<!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @product0(%[[ARRAY:.*]]: !hlfir.expr<42xi32>) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL]] to %[[MASK]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.product %[[ARRAY]] dim %[[C1]] mask %[[BOX]] : (!hlfir.expr<42xi32>, index, !fir.box<!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// array is an expression of assumed shape
func.func @product1(%arg0: !hlfir.expr<?xi32>) {
  %mask = fir.alloca !fir.logical<4>
  %c_1 = arith.constant 1 : index
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.product %arg0 dim %c_1 mask %mask_box : (!hlfir.expr<?xi32>, index, !fir.box<!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @product1(%[[ARRAY:.*]]: !hlfir.expr<?xi32>) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL:.*]] to %[[MASK:.*]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %[[MASK:.*]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.product %[[ARRAY:.*]] dim %[[C1]] mask %[[BOX]] : (!hlfir.expr<?xi32>, index, !fir.box<!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// boxed array
func.func @product2(%arg0: !fir.box<!fir.array<42xi32>>) {
  %mask = fir.alloca !fir.logical<4>
  %c_1 = arith.constant 1 : index
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.product %arg0 dim %c_1 mask %mask_box : (!fir.box<!fir.array<42xi32>>, index, !fir.box<!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @product2(%[[ARRAY:.*]]: !fir.box<!fir.array<42xi32>>) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL:.*]] to %[[MASK:.*]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %[[MASK:.*]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.product %[[ARRAY:.*]] dim %[[C1]] mask %[[BOX]] : (!fir.box<!fir.array<42xi32>>, index, !fir.box<!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// assumed shape boxed array
func.func @product3(%arg0: !fir.box<!fir.array<?xi32>>) {
  %mask = fir.alloca !fir.logical<4>
  %c_1 = arith.constant 1 : index
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.product %arg0 dim %c_1 mask %mask_box : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @product3(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL:.*]] to %[[MASK:.*]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %[[MASK:.*]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.product %[[ARRAY:.*]] dim %[[C1]] mask %[[BOX]] : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// known shape expr mask
func.func @product4(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !hlfir.expr<42x!fir.logical<4>>) {
  %c_1 = arith.constant 1 : index
  %0 = hlfir.product %arg0 dim %c_1 mask %arg1 : (!fir.box<!fir.array<?xi32>>, index, !hlfir.expr<42x!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @product4(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !hlfir.expr<42x!fir.logical<4>>) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   hlfir.product %[[ARRAY]] dim %[[C1]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, index, !hlfir.expr<42x!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// assumed shape expr mask
func.func @product5(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !hlfir.expr<?x!fir.logical<4>>) {
  %c_1 = arith.constant 1 : index
  %0 = hlfir.product %arg0 dim %c_1 mask %arg1 : (!fir.box<!fir.array<?xi32>>, index, !hlfir.expr<?x!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @product5(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !hlfir.expr<?x!fir.logical<4>>) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   hlfir.product %[[ARRAY]] dim %[[C1]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, index, !hlfir.expr<?x!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// known shape array mask
func.func @product6(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<42x!fir.logical<4>>>) {
  %c_1 = arith.constant 1 : index
  %0 = hlfir.product %arg0 dim %c_1 mask %arg1 : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.array<42x!fir.logical<4>>>) -> i32
  return
}
// CHECK:      func.func @product6(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !fir.box<!fir.array<42x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   hlfir.product %[[ARRAY]] dim %[[C1]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.array<42x!fir.logical<4>>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// assumed shape array mask
func.func @product7(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %c_1 = arith.constant 1 : index
  %0 = hlfir.product %arg0 dim %c_1 mask %arg1 : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
  return
}
// CHECK:      func.func @product7(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   hlfir.product %[[ARRAY]] dim %[[C1]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// known shape expr return
func.func @product8(%arg0: !fir.box<!fir.array<2x2xi32>>, %arg1: i32) {
  %mask = fir.alloca !fir.logical<4>
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.product %arg0 dim %arg1 mask %mask_box : (!fir.box<!fir.array<2x2xi32>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<2xi32>
  return
}
// CHECK:      func.func @product8(%[[ARRAY:.*]]: !fir.box<!fir.array<2x2xi32>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL]] to %[[MASK]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.product %[[ARRAY]] dim %[[DIM]] mask %[[BOX]] : (!fir.box<!fir.array<2x2xi32>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<2xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }


// assumed shape expr return
func.func @product9(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: i32) {
  %mask = fir.alloca !fir.logical<4>
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.product %arg0 dim %arg1 mask %mask_box : (!fir.box<!fir.array<?x?xi32>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?xi32>
  return
}
// CHECK:      func.func @product9(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL]] to %[[MASK]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.product %[[ARRAY]] dim %[[DIM]] mask %[[BOX]] : (!fir.box<!fir.array<?x?xi32>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.product with only an array argument
func.func @product10(%arg0: !fir.box<!fir.array<?x?xi32>>) {
  %product = hlfir.product %arg0 : (!fir.box<!fir.array<?x?xi32>>) -> i32
  return
}
// CHECK:      func.func @product10(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>
// CHECK-NEXT:   %[[PRODUCT:.*]] = hlfir.product %[[ARRAY]] : (!fir.box<!fir.array<?x?xi32>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.product with array and dim argument
func.func @product11(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: i32) {
  %product = hlfir.product %arg0 dim %arg1 : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
  return
}
// CHECK:      func.func @product11(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>, %[[DIM:.*]]: i32
// CHECK-NEXT:   %[[PRODUCT:.*]] = hlfir.product %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.product with array and mask argument
func.func @product12(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.logical<4>) {
  %product = hlfir.product %arg0 mask %arg1 : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> i32
  return
}
// CHECK:      func.func @product12(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !fir.logical<4>
// CHECK-NEXT:   %[[PRODUCT:.*]] = hlfir.product %[[ARRAY]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.product with dim argument with an unusual type
func.func @product13(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: index) {
  %product = hlfir.product %arg0 dim %arg1 : (!fir.box<!fir.array<?x?xi32>>, index) -> !hlfir.expr<?xi32>
  return
}
// CHECK:      func.func @product13(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>, %[[DIM:.*]]: index
// CHECK-NEXT:   %[[PRODUCT:.*]] = hlfir.product %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x?xi32>>, index) -> !hlfir.expr<?xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.product with mask argument of unusual type
func.func @product14(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: i1) {
  %product = hlfir.product %arg0 mask %arg1 : (!fir.box<!fir.array<?xi32>>, i1) -> i32
  return
}
// CHECK:      func.func @product14(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: i1
// CHECK-NEXT:   %[[PRODUCT:.*]] = hlfir.product %[[ARRAY]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, i1) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.product with mask argument of ref<array<>> type
func.func @product15(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.ref<!fir.array<?x!fir.logical<4>>>) {
  %product = hlfir.product %arg0 mask %arg1 : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?x!fir.logical<4>>>) -> i32
  return
}
// CHECK:      func.func @product15(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !fir.ref<!fir.array<?x!fir.logical<4>>>
// CHECK-NEXT:   %[[PRODUCT:.*]] = hlfir.product %[[ARRAY]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?x!fir.logical<4>>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
