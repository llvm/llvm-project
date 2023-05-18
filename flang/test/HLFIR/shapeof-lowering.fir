// Test hlfir.shape_of lowering
// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @shapeof_asexpr(%arg0: !fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.shape<1> {
  %c0 = arith.constant 0 : index
  %59:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  %60 = fir.box_addr %arg0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %61 = fir.shape_shift %59#0, %59#1 : (index, index) -> !fir.shapeshift<1>
  %62:2 = hlfir.declare %60(%61) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
  %true = arith.constant true
  %63 = hlfir.as_expr %62#0 move %true : (!fir.box<!fir.array<?xf32>>, i1) -> !hlfir.expr<?xf32>
  %64 = hlfir.shape_of %63 : (!hlfir.expr<?xf32>) -> !fir.shape<1>
  return %64 : !fir.shape<1>
}
// CHECK-LABEL: @shapeof_asexpr
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0
// CHECK-NEXT:    %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]]
// CHECK-NEXT:    %[[BOX_ADDR:.*]] = fir.box_addr %[[ARG0]]
// CHECK-NEXT:    %[[SHPE_SHFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// CHECK-NEXT:    %[[VAR:.*]]:2 = hlfir.declare %[[BOX_ADDR]](%[[SHPE_SHFT]])
// CHECK-NEXT:    %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:    %[[TUPLE0:.*]] = fir.undefined tuple
// CHECK-NEXT:    %[[TUPLE1:.*]] = fir.insert_value %[[TUPLE0]], %[[TRUE]]
// CHECK-NEXT:    %[[TUPLE2:.*]] = fir.insert_value %[[TUPLE1]], %[[VAR]]#0
// CHECK-NEXT:    %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1
// CHECK-NEXT:    return %[[SHAPE]]

func.func @shapeof_elemental() -> !fir.shape<1> {
  %c1 = arith.constant 1 : index
  %0 = fir.shape %c1 : (index) -> !fir.shape<1>
  %1 = hlfir.elemental %0 : (!fir.shape<1>) -> !hlfir.expr<?xindex> {
  ^bb0(%arg3: index):
    hlfir.yield_element %arg3 : index
  }
  %2 = hlfir.shape_of %1 : (!hlfir.expr<?xindex>) -> !fir.shape<1>
  return %2 : !fir.shape<1>
}
// CHECK-LABEL: @shapeof_elemental
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[SHAPE:.*]] = fir.shape %[[C1]]
// CHECK:         fir.do_loop %{{.*}} = %{{.*}} to %[[C1:.*]]
// CHECK:         return %[[SHAPE]]

func.func @shapeof_fallback(%arg0: !hlfir.expr<1x2x3xi32>) -> !fir.shape<3> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<1x2x3xi32>) -> !fir.shape<3>
  return %shape : !fir.shape<3>
}
// CHECK-LABEL: @shapeof_fallback
// CHECK:           %[[EXPR:.*]]: !hlfir.expr<1x2x3xi32>
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[C3:.*]] = arith.constant 3 : index
// CHECK-NEXT:    %[[SHAPE:.*]] = fir.shape %[[C1]], %[[C2]], %[[C3]] :
// CHECK-NEXT:    return %[[SHAPE]]
