// Test hlfir.pack simplification for scalar .TRUE. mask:
// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

func.func @pack_scalar_true_mask(%arg0: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32> {
  %true = arith.constant 1 : i1
  %mask = fir.convert %true : (i1) -> !fir.logical<4>
  %res = hlfir.pack %arg0 %mask : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @pack_scalar_true_mask(
// CHECK-SAME:                                   %[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM0:.*]]:3 = fir.box_dims %[[ARRAY]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:           %[[UNIT_SHAPE:.*]] = fir.shape %[[C1]] : (index) -> !fir.shape<1>
// CHECK:           %[[SHAPE_EXPR:.*]] = hlfir.elemental %[[UNIT_SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<1xindex> {
// CHECK:           ^bb0(%{{.*}}: index):
// CHECK:             hlfir.yield_element %{{.*}} : index
// CHECK:           }
// CHECK:           %[[SHAPE_VAL:.*]] = hlfir.apply %[[SHAPE_EXPR]], %[[C1]] : (!hlfir.expr<1xindex>, index) -> index
// CHECK:           %[[RESULT_SHAPE:.*]] = fir.shape %[[SHAPE_VAL]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL:.*]] = hlfir.elemental %[[RESULT_SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
// CHECK:           ^bb0(%[[IDX:.*]]: index):
// CHECK:             %[[ARRAY_DIM0:.*]]:3 = fir.box_dims %[[ARRAY]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:             %[[LB_ADJ:.*]] = arith.subi %[[ARRAY_DIM0]]#0, %[[C1]] overflow<nuw> : index
// CHECK:             %[[ARRAY_IDX:.*]] = arith.addi %[[IDX]], %[[LB_ADJ]] overflow<nuw> : index
// CHECK:             %[[ELEM_ADDR:.*]] = hlfir.designate %[[ARRAY]] (%[[ARRAY_IDX]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[ELEM:.*]] = fir.load %[[ELEM_ADDR]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[ELEM]] : i32
// CHECK:           }
// CHECK:           hlfir.destroy %[[SHAPE_EXPR]] : !hlfir.expr<1xindex>
// CHECK-NOT:       fir.alloca
// CHECK-NOT:       hlfir.pack
// CHECK-NOT:       hlfir.reshape
// CHECK:           return %[[ELEMENTAL]] : !hlfir.expr<?xi32>
// CHECK:         }

func.func @pack_scalar_true_mask_2d(%arg0: !fir.box<!fir.array<?x?xi32>>) -> !hlfir.expr<?xi32> {
  %true = arith.constant 1 : i1
  %mask = fir.convert %true : (i1) -> !fir.logical<4>
  %res = hlfir.pack %arg0 %mask : (!fir.box<!fir.array<?x?xi32>>, !fir.logical<4>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @pack_scalar_true_mask_2d(
// CHECK-SAME:                                     %[[ARRAY:.*]]: !fir.box<!fir.array<?x?xi32>>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM0:.*]]:3 = fir.box_dims %[[ARRAY]], %[[C0]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[DIM1:.*]]:3 = fir.box_dims %[[ARRAY]], %[[C1]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[PROD:.*]] = arith.muli %[[DIM0]]#1, %[[DIM1]]#1 overflow<nuw> : index
// CHECK:           %[[UNIT_SHAPE:.*]] = fir.shape %[[C1]] : (index) -> !fir.shape<1>
// CHECK:           %[[SHAPE_EXPR:.*]] = hlfir.elemental %[[UNIT_SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<1xindex> {
// CHECK:           ^bb0(%{{.*}}: index):
// CHECK:             hlfir.yield_element %[[PROD]] : index
// CHECK:           }
// CHECK:           %[[HOISTED_DIM0:.*]]:3 = fir.box_dims %[[ARRAY]], %[[C0]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[SHAPE_VAL:.*]] = hlfir.apply %[[SHAPE_EXPR]], %[[C1]] : (!hlfir.expr<1xindex>, index) -> index
// CHECK:           %[[RESULT_SHAPE:.*]] = fir.shape %[[SHAPE_VAL]] : (index) -> !fir.shape<1>
// CHECK:           %[[ELEMENTAL:.*]] = hlfir.elemental %[[RESULT_SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
// CHECK:           ^bb0(%[[IDX:.*]]: index):
// CHECK:             %[[ZERO_IDX:.*]] = arith.subi %[[IDX]], %[[C1]] overflow<nuw> : index
// CHECK:             %[[IDX0:.*]] = arith.remui %[[ZERO_IDX]], %[[HOISTED_DIM0]]#1 : index
// CHECK:             %[[IDX1:.*]] = arith.divui %[[ZERO_IDX]], %[[HOISTED_DIM0]]#1 : index
// CHECK:             %[[ONE_IDX0:.*]] = arith.addi %[[IDX0]], %[[C1]] overflow<nuw> : index
// CHECK:             %[[ONE_IDX1:.*]] = arith.addi %[[IDX1]], %[[C1]] overflow<nuw> : index
// CHECK:             %[[ARRAY_DIM0:.*]]:3 = fir.box_dims %[[ARRAY]], %[[C0]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[ARRAY_DIM1:.*]]:3 = fir.box_dims %[[ARRAY]], %[[C1]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:             %[[LB0:.*]] = arith.subi %[[ARRAY_DIM0]]#0, %[[C1]] overflow<nuw> : index
// CHECK:             %[[ARRAY_IDX0:.*]] = arith.addi %[[ONE_IDX0]], %[[LB0]] overflow<nuw> : index
// CHECK:             %[[LB1:.*]] = arith.subi %[[ARRAY_DIM1]]#0, %[[C1]] overflow<nuw> : index
// CHECK:             %[[ARRAY_IDX1:.*]] = arith.addi %[[ONE_IDX1]], %[[LB1]] overflow<nuw> : index
// CHECK:             %[[ELEM_ADDR:.*]] = hlfir.designate %[[ARRAY]] (%[[ARRAY_IDX0]], %[[ARRAY_IDX1]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[ELEM:.*]] = fir.load %[[ELEM_ADDR]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[ELEM]] : i32
// CHECK:           }
// CHECK:           hlfir.destroy %[[SHAPE_EXPR]] : !hlfir.expr<1xindex>
// CHECK-NOT:       fir.alloca
// CHECK-NOT:       hlfir.pack
// CHECK-NOT:       hlfir.reshape
// CHECK:           return %[[ELEMENTAL]] : !hlfir.expr<?xi32>
// CHECK:         }

func.func @pack_var_mask(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.ref<!fir.logical<4>>) -> !hlfir.expr<?xi32> {
  %mask = fir.load %arg1 : !fir.ref<!fir.logical<4>>
  %res = hlfir.pack %arg0 %mask : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @pack_var_mask(
// CHECK-SAME:                            %[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:                            %[[MASK:.*]]: !fir.ref<!fir.logical<4>>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[MASK_VAL:.*]] = fir.load %[[MASK]] : !fir.ref<!fir.logical<4>>
// CHECK:           %[[PACK:.*]] = hlfir.pack %[[ARRAY]] %[[MASK_VAL]] : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>) -> !hlfir.expr<?xi32>
// CHECK-NOT:       hlfir.reshape
// CHECK:           return %[[PACK]] : !hlfir.expr<?xi32>
// CHECK:         }

func.func @pack_with_vector(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32> {
  %true = arith.constant 1 : i1
  %mask = fir.convert %true : (i1) -> !fir.logical<4>
  %res = hlfir.pack %arg0 %mask vector %arg1 : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
  return %res : !hlfir.expr<?xi32>
}
// CHECK-LABEL:   func.func @pack_with_vector(
// CHECK-SAME:                               %[[ARRAY:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:                               %[[VECTOR:.*]]: !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32> {
// CHECK:           %[[PACK:.*]] = hlfir.pack %[[ARRAY]] {{.*}} vector %[[VECTOR]] : (!fir.box<!fir.array<?xi32>>, !fir.logical<4>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
// CHECK-NOT:       hlfir.reshape
// CHECK:           return %[[PACK]] : !hlfir.expr<?xi32>
// CHECK:         }
