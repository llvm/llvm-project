// Test hlfir.reshape operation parse, verify (no errors), and unparse
// RUN: fir-opt %s | fir-opt | FileCheck %s

// Operands are expressions of known shape
func.func @reshape1(%arg0: !hlfir.expr<42xi32>, %arg1: !hlfir.expr<1xi32>) -> !hlfir.expr<?xi32> {
  %0 = hlfir.reshape %arg0 %arg1 : (!hlfir.expr<42xi32>, !hlfir.expr<1xi32>) -> !hlfir.expr<?xi32>
  return %0 : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @reshape1(
// CHECK-SAME:                        %[[VAL_0:.*]]: !hlfir.expr<42xi32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: !hlfir.expr<1xi32>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[VAL_2:.*]] = hlfir.reshape %[[VAL_0]] %[[VAL_1]] : (!hlfir.expr<42xi32>, !hlfir.expr<1xi32>) -> !hlfir.expr<?xi32>
// CHECK:           return %[[VAL_2]] : !hlfir.expr<?xi32>
// CHECK:         }

// Operands are expressions of assumed shape
func.func @reshape2(%arg0: !hlfir.expr<?xi32>, %arg1: !hlfir.expr<1xi32>) -> !hlfir.expr<4xi32> {
  %0 = hlfir.reshape %arg0 %arg1 pad %arg0 order %arg0 : (!hlfir.expr<?xi32>, !hlfir.expr<1xi32>, !hlfir.expr<?xi32>, !hlfir.expr<?xi32>) -> !hlfir.expr<4xi32>
  return %0 : !hlfir.expr<4xi32>
}
// CHECK-LABEL:   func.func @reshape2(
// CHECK-SAME:                        %[[VAL_0:.*]]: !hlfir.expr<?xi32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: !hlfir.expr<1xi32>) -> !hlfir.expr<4xi32> {
// CHECK:           %[[VAL_2:.*]] = hlfir.reshape %[[VAL_0]] %[[VAL_1]] pad %[[VAL_0]] order %[[VAL_0]] : (!hlfir.expr<?xi32>, !hlfir.expr<1xi32>, !hlfir.expr<?xi32>, !hlfir.expr<?xi32>) -> !hlfir.expr<4xi32>
// CHECK:           return %[[VAL_2]] : !hlfir.expr<4xi32>
// CHECK:         }

// Operands are boxed array
func.func @reshape3(%arg0: !fir.box<!fir.array<42xi32>>, %arg1: !fir.box<!fir.array<2xi32>>) -> !hlfir.expr<?x?xi32> {
  %0 = hlfir.reshape %arg0 %arg1 : (!fir.box<!fir.array<42xi32>>, !fir.box<!fir.array<2xi32>>) -> !hlfir.expr<?x?xi32>
  return %0 : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @reshape3(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<42xi32>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: !fir.box<!fir.array<2xi32>>) -> !hlfir.expr<?x?xi32> {
// CHECK:           %[[VAL_2:.*]] = hlfir.reshape %[[VAL_0]] %[[VAL_1]] : (!fir.box<!fir.array<42xi32>>, !fir.box<!fir.array<2xi32>>) -> !hlfir.expr<?x?xi32>
// CHECK:           return %[[VAL_2]] : !hlfir.expr<?x?xi32>
// CHECK:         }

// Operands are assumed shape boxed arrays
func.func @reshape4(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<1xi32>>) -> !hlfir.expr<?xi32> {
  %0 = hlfir.reshape %arg0 %arg1 pad %arg0 order %arg0 : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<1xi32>>, !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
  return %0 : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @reshape4(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: !fir.box<!fir.array<1xi32>>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[VAL_2:.*]] = hlfir.reshape %[[VAL_0]] %[[VAL_1]] pad %[[VAL_0]] order %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<1xi32>>, !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
// CHECK:           return %[[VAL_2]] : !hlfir.expr<?xi32>
// CHECK:         }

// Operands are ref<array<>> of known shape
func.func @reshape5(%arg0: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?xi32> {
  %0 = hlfir.reshape %arg0 %arg0 pad %arg0 order %arg0 : (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?xi32>
  return %0 : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @reshape5(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[VAL_1:.*]] = hlfir.reshape %[[VAL_0]] %[[VAL_0]] pad %[[VAL_0]] order %[[VAL_0]] : (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?xi32>
// CHECK:           return %[[VAL_1]] : !hlfir.expr<?xi32>
// CHECK:         }

// Polymorphic operands
func.func @reshape6(%arg0: !fir.class<!fir.heap<!fir.array<?x!fir.type<whatever>>>>, %arg1: !hlfir.expr<1xi32>) -> !hlfir.expr<?x!fir.type<whatever>?> {
  %0 = hlfir.reshape %arg0 %arg1 pad %arg0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<whatever>>>>, !hlfir.expr<1xi32>, !fir.class<!fir.heap<!fir.array<?x!fir.type<whatever>>>>) -> !hlfir.expr<?x!fir.type<whatever>?>
  return %0 : !hlfir.expr<?x!fir.type<whatever>?>
}
// CHECK-LABEL:   func.func @reshape6(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.class<!fir.heap<!fir.array<?x!fir.type<whatever>>>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: !hlfir.expr<1xi32>) -> !hlfir.expr<?x!fir.type<whatever>?> {
// CHECK:           %[[VAL_2:.*]] = hlfir.reshape %[[VAL_0]] %[[VAL_1]] pad %[[VAL_0]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<whatever>>>>, !hlfir.expr<1xi32>, !fir.class<!fir.heap<!fir.array<?x!fir.type<whatever>>>>) -> !hlfir.expr<?x!fir.type<whatever>?>
// CHECK:           return %[[VAL_2]] : !hlfir.expr<?x!fir.type<whatever>?>
// CHECK:         }

// Allow character LEN mismatch for ARRAY/PAD and the result
func.func @reshape7(%arg0: !hlfir.expr<1x!fir.char<2,2>>, %arg1: !hlfir.expr<1xi32>, %arg2: !hlfir.expr<1x!fir.char<2,1>>) -> !hlfir.expr<?x!fir.char<2,3>> {
  %0 = hlfir.reshape %arg0 %arg1 pad %arg2 : (!hlfir.expr<1x!fir.char<2,2>>, !hlfir.expr<1xi32>, !hlfir.expr<1x!fir.char<2,1>>) -> !hlfir.expr<?x!fir.char<2,3>>
  return %0 : !hlfir.expr<?x!fir.char<2,3>>
}
// CHECK-LABEL:   func.func @reshape7(
// CHECK-SAME:                        %[[VAL_0:.*]]: !hlfir.expr<1x!fir.char<2,2>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: !hlfir.expr<1xi32>,
// CHECK-SAME:                        %[[VAL_2:.*]]: !hlfir.expr<1x!fir.char<2>>) -> !hlfir.expr<?x!fir.char<2,3>> {
// CHECK:           %[[VAL_3:.*]] = hlfir.reshape %[[VAL_0]] %[[VAL_1]] pad %[[VAL_2]] : (!hlfir.expr<1x!fir.char<2,2>>, !hlfir.expr<1xi32>, !hlfir.expr<1x!fir.char<2>>) -> !hlfir.expr<?x!fir.char<2,3>>
// CHECK:           return %[[VAL_3]] : !hlfir.expr<?x!fir.char<2,3>>
// CHECK:         }
