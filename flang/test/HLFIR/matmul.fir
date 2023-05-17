// Test hlfir.matmul operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

// arguments are expressions of known shape
func.func @matmul0(%arg0: !hlfir.expr<2x2xi32>, %arg1: !hlfir.expr<2x2xi32>) {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<2x2xi32>, !hlfir.expr<2x2xi32>) -> !hlfir.expr<2x2xi32>
  return
}
// CHECK-LABEL: func.func @matmul0
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<2x2xi32>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<2x2xi32>) {
// CHECK-NEXT:    %[[RES:.*]] = hlfir.matmul %[[ARG0]] %[[ARG1]] : (!hlfir.expr<2x2xi32>, !hlfir.expr<2x2xi32>) -> !hlfir.expr<2x2xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// arguments are expressions of assumed shape
func.func @matmul1(%arg0: !hlfir.expr<?x?xi32>, %arg1: !hlfir.expr<?x?xi32>) {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xi32>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  return
}
// CHECK-LABEL: func.func @matmul1
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<?x?xi32>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<?x?xi32>) {
// CHECK-NEXT:    %[[RES:.*]] = hlfir.matmul %[[ARG0]] %[[ARG1]] : (!hlfir.expr<?x?xi32>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// arguments are expressions where only some dimensions are known #1
func.func @matmul2(%arg0: !hlfir.expr<2x?xi32>, %arg1: !hlfir.expr<?x2xi32>) {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<2x?xi32>, !hlfir.expr<?x2xi32>) -> !hlfir.expr<2x2xi32>
  return
}
// CHECK-LABEL: func.func @matmul2
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<2x?xi32>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<?x2xi32>) {
// CHECK-NEXT:    %[[RES:.*]] = hlfir.matmul %[[ARG0]] %[[ARG1]] : (!hlfir.expr<2x?xi32>, !hlfir.expr<?x2xi32>) -> !hlfir.expr<2x2xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// arguments are expressions where only some dimensions are known #2
func.func @matmul3(%arg0: !hlfir.expr<?x2xi32>, %arg1: !hlfir.expr<2x?xi32>) {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x2xi32>, !hlfir.expr<2x?xi32>) -> !hlfir.expr<?x?xi32>
  return
}
// CHECK-LABEL: func.func @matmul3
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<?x2xi32>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<2x?xi32>) {
// CHECK-NEXT:    %[[RES:.*]] = hlfir.matmul %[[ARG0]] %[[ARG1]] : (!hlfir.expr<?x2xi32>, !hlfir.expr<2x?xi32>) -> !hlfir.expr<?x?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// arguments are logicals
func.func @matmul4(%arg0: !hlfir.expr<?x?x!fir.logical<4>>, %arg1: !hlfir.expr<?x?x!fir.logical<4>>) {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?x!fir.logical<4>>, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?x?x!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @matmul4
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<?x?x!fir.logical<4>>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<?x?x!fir.logical<4>>) {
// CHECK-NEXT:    %[[RES:.*]] = hlfir.matmul %[[ARG0]] %[[ARG1]] : (!hlfir.expr<?x?x!fir.logical<4>>, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<?x?x!fir.logical<4>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// lhs is rank 1
func.func @matmul5(%arg0: !hlfir.expr<?xi32>, %arg1: !hlfir.expr<?x?xi32>) {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?xi32>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @matmul5
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<?xi32>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<?x?xi32>) {
// CHECK-NEXT:    %[[RES:.*]] = hlfir.matmul %[[ARG0]] %[[ARG1]] : (!hlfir.expr<?xi32>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// rhs is rank 1
func.func @matmul6(%arg0: !hlfir.expr<?x?xi32>, %arg1: !hlfir.expr<?xi32>) {
  %res = hlfir.matmul %arg0 %arg1 : (!hlfir.expr<?x?xi32>, !hlfir.expr<?xi32>) -> !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @matmul6
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<?x?xi32>,
// CHECK:           %[[ARG1:.*]]: !hlfir.expr<?xi32>) {
// CHECK-NEXT:    %[[RES:.*]] = hlfir.matmul %[[ARG0]] %[[ARG1]] : (!hlfir.expr<?x?xi32>, !hlfir.expr<?xi32>) -> !hlfir.expr<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// arguments are boxed arrays
func.func @matmul7(%arg0: !fir.box<!fir.array<2x2xf32>>, %arg1: !fir.box<!fir.array<2x2xf32>>) {
  %res = hlfir.matmul %arg0 %arg1 : (!fir.box<!fir.array<2x2xf32>>, !fir.box<!fir.array<2x2xf32>>) -> !hlfir.expr<2x2xf32>
  return
}
// CHECK-LABEL: func.func @matmul7
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<2x2xf32>>,
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<2x2xf32>>) {
// CHECK-NEXT:    %[[RES:.*]] = hlfir.matmul %[[ARG0]] %[[ARG1]] : (!fir.box<!fir.array<2x2xf32>>, !fir.box<!fir.array<2x2xf32>>) -> !hlfir.expr<2x2xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
