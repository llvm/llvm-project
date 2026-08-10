// Test hlfir.transpose operation parse, verify (no errors), and unparse

// RUN: fir-opt %s | fir-opt | FileCheck %s

// square matrix of known shape
func.func @transpose0(%arg0: !hlfir.expr<2x2xi32>) {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<2x2xi32>) -> !hlfir.expr<2x2xi32>
  return
}
// CHECK-LABEL: func.func @transpose0
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<2x2xi32>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.transpose %[[ARG0]] : (!hlfir.expr<2x2xi32>) -> !hlfir.expr<2x2xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// rectangular matrix of known shape
func.func @transpose1(%arg0: !hlfir.expr<2x3xi32>) {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<2x3xi32>) -> !hlfir.expr<3x2xi32>
  return
}
// CHECK-LABEL: func.func @transpose1
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<2x3xi32>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.transpose %[[ARG0]] : (!hlfir.expr<2x3xi32>) -> !hlfir.expr<3x2xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// matrix of assumed shape
func.func @transpose2(%arg0: !hlfir.expr<?x?xi32>) {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
  return
}
// CHECK-LABEL: func.func @transpose2
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<?x?xi32>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.transpose %[[ARG0]] : (!hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// matrix where only some dimensions are known #1
func.func @transpose3(%arg0: !hlfir.expr<?x2xi32>) {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<?x2xi32>) -> !hlfir.expr<2x?xi32>
  return
}
// CHECK-LABEL: func.func @transpose3
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<?x2xi32>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.transpose %[[ARG0]] : (!hlfir.expr<?x2xi32>) -> !hlfir.expr<2x?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// matrix where only some dimensions are known #2
func.func @transpose4(%arg0: !hlfir.expr<2x?xi32>) {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<2x?xi32>) -> !hlfir.expr<?x2xi32>
  return
}
// CHECK-LABEL: func.func @transpose4
// CHECK:           %[[ARG0:.*]]: !hlfir.expr<2x?xi32>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.transpose %[[ARG0]] : (!hlfir.expr<2x?xi32>) -> !hlfir.expr<?x2xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// matrix is a boxed array
func.func @transpose5(%arg0: !fir.box<!fir.array<1x2xi32>>) {
  %res = hlfir.transpose %arg0 : (!fir.box<!fir.array<1x2xi32>>) -> !hlfir.expr<2x1xi32>
  return
}
// CHECK-LABEL: func.func @transpose5
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<1x2xi32>>
// CHECK-NEXT:    %[[RES:.*]] = hlfir.transpose %[[ARG0]] : (!fir.box<!fir.array<1x2xi32>>) -> !hlfir.expr<2x1xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
