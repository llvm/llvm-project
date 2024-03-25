// Test hlfir.dot_product operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

// arguments are expressions of known shape
func.func @dot_product0(%arg0: !hlfir.expr<2xi32>, %arg1: !hlfir.expr<2xi32>) {
  %res = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<2xi32>, !hlfir.expr<2xi32>) -> i32
  return
}
// CHECK-LABEL: func.func @dot_product0
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<2xi32>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<2xi32>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.dot_product %[[ARG0]] %[[ARG1]] : (!hlfir.expr<2xi32>, !hlfir.expr<2xi32>) -> i32
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// arguments are expressions of assumed shape
func.func @dot_product1(%arg0: !hlfir.expr<?xi32>, %arg1: !hlfir.expr<?xi32>) {
  %res = hlfir.dot_product %arg0 %arg1 : (!hlfir.expr<?xi32>, !hlfir.expr<?xi32>) -> i32
  return
}
// CHECK-LABEL: func.func @dot_product1
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<?xi32>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<?xi32>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.dot_product %[[ARG0]] %[[ARG1]] : (!hlfir.expr<?xi32>, !hlfir.expr<?xi32>) -> i32
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// arguments are boxed arrays
func.func @dot_product2(%arg0: !fir.box<!fir.array<2xi32>>, %arg1: !fir.box<!fir.array<2xi32>>) {
  %res = hlfir.dot_product %arg0 %arg1 : (!fir.box<!fir.array<2xi32>>, !fir.box<!fir.array<2xi32>>) -> i32
  return
}
// CHECK-LABEL: func.func @dot_product2
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<2xi32>>,
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<2xi32>>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.dot_product %[[ARG0]] %[[ARG1]] : (!fir.box<!fir.array<2xi32>>, !fir.box<!fir.array<2xi32>>) -> i32
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// arguments are logical
func.func @dot_product3(%arg0: !fir.box<!fir.array<2x!fir.logical<4>>>, %arg1: !fir.box<!fir.array<2x!fir.logical<4>>>) {
  %res = hlfir.dot_product %arg0 %arg1 : (!fir.box<!fir.array<2x!fir.logical<4>>>, !fir.box<!fir.array<2x!fir.logical<4>>>) -> !fir.logical<4>
  return
}
// CHECK-LABEL: func.func @dot_product3
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<2x!fir.logical<4>>>,
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<2x!fir.logical<4>>>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.dot_product %[[ARG0]] %[[ARG1]] : (!fir.box<!fir.array<2x!fir.logical<4>>>, !fir.box<!fir.array<2x!fir.logical<4>>>) -> !fir.logical<4>
// CHECK-NEXT:    return
// CHECK-NEXT:  }