// Test hlfir.minloc operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

// array is an expression of known shape
func.func @minloc0(%arg0: !hlfir.expr<42xi32>) {
  %mask = fir.alloca !fir.logical<4>
  %c_1 = arith.constant 1 : index
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.minloc %arg0 dim %c_1 mask %mask_box : (!hlfir.expr<42xi32>, index, !fir.box<!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @minloc0(%[[ARRAY:.*]]: !hlfir.expr<42xi32>) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL]] to %[[MASK]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.minloc %[[ARRAY]] dim %[[C1]] mask %[[BOX]] : (!hlfir.expr<42xi32>, index, !fir.box<!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// array is an expression of assumed shape
func.func @minloc1(%arg0: !hlfir.expr<?xi32>) {
  %mask = fir.alloca !fir.logical<4>
  %c_1 = arith.constant 1 : index
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.minloc %arg0 dim %c_1 mask %mask_box : (!hlfir.expr<?xi32>, index, !fir.box<!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @minloc1(%[[ARRAY:.*]]: !hlfir.expr<?xi32>) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL:.*]] to %[[MASK:.*]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %[[MASK:.*]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.minloc %[[ARRAY:.*]] dim %[[C1]] mask %[[BOX]] : (!hlfir.expr<?xi32>, index, !fir.box<!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// boxed array
func.func @minloc2(%arg0: !fir.box<!fir.array<42xi32>>) {
  %mask = fir.alloca !fir.logical<4>
  %c_1 = arith.constant 1 : index
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.minloc %arg0 dim %c_1 mask %mask_box : (!fir.box<!fir.array<42xi32>>, index, !fir.box<!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @minloc2(%[[ARRAY:.*]]: !fir.box<!fir.array<42xi32>>) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL:.*]] to %[[MASK:.*]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %[[MASK:.*]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.minloc %[[ARRAY:.*]] dim %[[C1]] mask %[[BOX]] : (!fir.box<!fir.array<42xi32>>, index, !fir.box<!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// assumed shape boxed array
func.func @minloc3(%arg0: !fir.box<!fir.array<?xi32>>) {
  %mask = fir.alloca !fir.logical<4>
  %c_1 = arith.constant 1 : index
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.minloc %arg0 dim %c_1 mask %mask_box : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @minloc3(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL:.*]] to %[[MASK:.*]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %[[MASK:.*]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.minloc %[[ARRAY:.*]] dim %[[C1]] mask %[[BOX]] : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// known shape expr mask
func.func @minloc4(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !hlfir.expr<42x!fir.logical<4>>) {
  %c_1 = arith.constant 1 : index
  %0 = hlfir.minloc %arg0 dim %c_1 mask %arg1 : (!fir.box<!fir.array<?xi32>>, index, !hlfir.expr<42x!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @minloc4(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !hlfir.expr<42x!fir.logical<4>>) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   hlfir.minloc %[[ARRAY]] dim %[[C1]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, index, !hlfir.expr<42x!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// assumed shape expr mask
func.func @minloc5(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !hlfir.expr<?x!fir.logical<4>>) {
  %c_1 = arith.constant 1 : index
  %0 = hlfir.minloc %arg0 dim %c_1 mask %arg1 : (!fir.box<!fir.array<?xi32>>, index, !hlfir.expr<?x!fir.logical<4>>) -> i32
  return
}
// CHECK:      func.func @minloc5(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !hlfir.expr<?x!fir.logical<4>>) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   hlfir.minloc %[[ARRAY]] dim %[[C1]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, index, !hlfir.expr<?x!fir.logical<4>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// known shape array mask
func.func @minloc6(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<42x!fir.logical<4>>>) {
  %c_1 = arith.constant 1 : index
  %0 = hlfir.minloc %arg0 dim %c_1 mask %arg1 : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.array<42x!fir.logical<4>>>) -> i32
  return
}
// CHECK:      func.func @minloc6(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !fir.box<!fir.array<42x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   hlfir.minloc %[[ARRAY]] dim %[[C1]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.array<42x!fir.logical<4>>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// assumed shape array mask
func.func @minloc7(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %c_1 = arith.constant 1 : index
  %0 = hlfir.minloc %arg0 dim %c_1 mask %arg1 : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
  return
}
// CHECK:      func.func @minloc7(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   hlfir.minloc %[[ARRAY]] dim %[[C1]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, index, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
// CHECK-NEXT:   return
// CHECK-NEXT: }

// known shape expr return
func.func @minloc8(%arg0: !fir.box<!fir.array<2x2xi32>>, %arg1: i32) {
  %mask = fir.alloca !fir.logical<4>
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.minloc %arg0 dim %arg1 mask %mask_box : (!fir.box<!fir.array<2x2xi32>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<2xi32>
  return
}
// CHECK:      func.func @minloc8(%[[ARRAY:.*]]: !fir.box<!fir.array<2x2xi32>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL]] to %[[MASK]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.minloc %[[ARRAY]] dim %[[DIM]] mask %[[BOX]] : (!fir.box<!fir.array<2x2xi32>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<2xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// assumed shape expr return
func.func @minloc9(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: i32) {
  %mask = fir.alloca !fir.logical<4>
  %true = arith.constant true
  %true_logical = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %true_logical to %mask : !fir.ref<!fir.logical<4>>
  %mask_box = fir.embox %mask : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %0 = hlfir.minloc %arg0 dim %arg1 mask %mask_box : (!fir.box<!fir.array<?x?xi32>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?xi32>
  return
}
// CHECK:      func.func @minloc9(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>, %[[DIM:.*]]: i32) {
// CHECK-NEXT:   %[[MASK:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:   %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:   fir.store %[[LOGICAL]] to %[[MASK]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOX:.*]] = fir.embox %0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:   hlfir.minloc %[[ARRAY]] dim %[[DIM]] mask %[[BOX]] : (!fir.box<!fir.array<?x?xi32>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.minloc with only an array argument
func.func @minloc10(%arg0: !fir.box<!fir.array<?x?xi32>>) {
  %minloc = hlfir.minloc %arg0 : (!fir.box<!fir.array<?x?xi32>>) -> !hlfir.expr<1xi32>
  return
}
// CHECK:      func.func @minloc10(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] : (!fir.box<!fir.array<?x?xi32>>) -> !hlfir.expr<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.minloc with only a character array argument
func.func @minloc11(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>>) {
  %minloc = hlfir.minloc %arg0 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !hlfir.expr<1xi32>
  return
}
// CHECK:      func.func @minloc11(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>) -> !hlfir.expr<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.minloc with array and dim argument
func.func @minloc12(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: i32) {
  %minloc = hlfir.minloc %arg0 dim %arg1 : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
  return
}
// CHECK:      func.func @minloc12(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>, %[[DIM:.*]]: i32
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.minloc with array and mask argument
func.func @minloc13(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.logical<4>) {
  %minloc = hlfir.minloc %arg0 mask %arg1 : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<1xi32>
  return
}
// CHECK:      func.func @minloc13(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !fir.logical<4>
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.minloc with dim argument with an unusual type
func.func @minloc14(%arg0: !fir.box<!fir.array<?x?xi32>>, %arg1: index) {
  %minloc = hlfir.minloc %arg0 dim %arg1 : (!fir.box<!fir.array<?x?xi32>>, index) -> !hlfir.expr<?xi32>
  return
}
// CHECK:      func.func @minloc14(%[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>, %[[DIM:.*]]: index
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] dim %[[DIM]] : (!fir.box<!fir.array<?x?xi32>>, index) -> !hlfir.expr<?xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.minloc with mask argument of unusual type
func.func @minloc15(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: i1) {
  %minloc = hlfir.minloc %arg0 mask %arg1 : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<1xi32>
  return
}
// CHECK:      func.func @minloc15(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: i1
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.minloc with mask argument of ref<array<>> type
func.func @minloc16(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.ref<!fir.array<?x!fir.logical<4>>>) {
  %minloc = hlfir.minloc %arg0 mask %arg1 : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?x!fir.logical<4>>>) -> !hlfir.expr<1xi32>
  return
}
// CHECK:      func.func @minloc16(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: !fir.ref<!fir.array<?x!fir.logical<4>>>
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?x!fir.logical<4>>>) -> !hlfir.expr<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }


// hlfir.minloc with kind implied by the return type
func.func @minloc17(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: i1) {
  %minloc = hlfir.minloc %arg0 mask %arg1 : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<1xi16>
  return
}
// CHECK:      func.func @minloc17(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: i1
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] mask %[[MASK]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<1xi16>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// hlfir.minloc with back argument
func.func @minloc18(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: i1) {
  %true = arith.constant true
  %minloc = hlfir.minloc %arg0 mask %arg1 back %true : (!fir.box<!fir.array<?xi32>>, i1, i1) -> !hlfir.expr<1xi32>
  return
}
// CHECK:      func.func @minloc18(%[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>, %[[MASK:.*]]: i1
// CHECK-NEXT:   %[[C2:.*]] = arith.constant true
// CHECK-NEXT:   %[[minloc:.*]] = hlfir.minloc %[[ARRAY]] mask %[[MASK]] back %[[C2]] : (!fir.box<!fir.array<?xi32>>, i1, i1) -> !hlfir.expr<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }