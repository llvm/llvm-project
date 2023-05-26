// Test hlfir.shape_of operation parse, verify (no errors), and unparse
// RUN: fir-opt %s | fir-opt | FileCheck --check-prefix CHECK --check-prefix CHECK-ALL %s

// Test canonicalization
// RUN: fir-opt %s --canonicalize | FileCheck --check-prefix CHECK-CANON --check-prefix CHECK-ALL %s

func.func @shapeof(%arg0: !hlfir.expr<2x2xi32>) -> !fir.shape<2> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<2x2xi32>) -> !fir.shape<2>
  return %shape : !fir.shape<2>
}
// CHECK-ALL-LABEL: func.func @shapeof
// CHECK-ALL:          %[[EXPR:.*]]: !hlfir.expr<2x2xi32>

// CHECK-NEXT:       %[[SHAPE:.*]] = hlfir.shape_of %[[EXPR]] : (!hlfir.expr<2x2xi32>) -> !fir.shape<2>

// CHECK-CANON-NEXT: %[[C2:.*]] = arith.constant 2 : index
// CHECK-CANON-NEXT: %[[SHAPE:.*]] = fir.shape %[[C2]], %[[C2]] : (index, index) -> !fir.shape<2>

// CHECK-ALL-NEXT:   return %[[SHAPE]]

// no canonicalization of expressions with unknown extents
func.func @shapeof2(%arg0: !hlfir.expr<?x2xi32>) -> !fir.shape<2> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?x2xi32>) -> !fir.shape<2>
  return %shape : !fir.shape<2>
}
// CHECK-ALL-LABEL: func.func @shapeof2
// CHECK-ALL:          %[[EXPR:.*]]: !hlfir.expr<?x2xi32>
// CHECK-ALL-NEXT:   %[[SHAPE:.*]] = hlfir.shape_of %[[EXPR]] : (!hlfir.expr<?x2xi32>) -> !fir.shape<2>
// CHECK-ALL-NEXT:   return %[[SHAPE]]
