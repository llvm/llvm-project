// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

// box with known extents
func.func @transpose0(%arg0: !fir.box<!fir.array<1x2xi32>>) {
  %res = hlfir.transpose %arg0 : (!fir.box<!fir.array<1x2xi32>>) -> !hlfir.expr<2x1xi32>
  return
}
// CHECK-LABEL:   func.func @transpose0(
// CHECK-SAME:                          %[[ARG0:.*]]: !fir.box<!fir.array<1x2xi32>>) {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[C2]], %[[C1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[EXPR:.*]] = hlfir.elemental %[[SHAPE]] : (!fir.shape<2>) -> !hlfir.expr<2x1xi32> {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<1x2xi32>>, index) -> (index, index, index)
// CHECK:             %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:             %[[DIMS1:.*]]:3 = fir.box_dims %[[ARG0]], %[[C1_1]] : (!fir.box<!fir.array<1x2xi32>>, index) -> (index, index, index)
// CHECK:             %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:             %[[LOWER_BOUND0:.*]] = arith.subi %[[DIMS0]]#0, %[[C1_2]] : index
// CHECK:             %[[J_OFFSET:.*]] = arith.addi %[[J]], %[[LOWER_BOUND0]] : index
// CHECK:             %[[LOWER_BOUND1:.*]] = arith.subi %[[DIMS1]]#0, %[[C1_2]] : index
// CHECK:             %[[I_OFFSET:.*]] = arith.addi %[[I]], %[[LOWER_BOUND1]] : index
// CHECK:             %[[ELEMENT_REF:.*]] = hlfir.designate %[[ARG0]] (%[[J_OFFSET]], %[[I_OFFSET]])  : (!fir.box<!fir.array<1x2xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[ELEMENT:.*]] = fir.load %[[ELEMENT_REF]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[ELEMENT]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// expr with known extents
func.func @transpose1(%arg0: !hlfir.expr<1x2xi32>) {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<1x2xi32>) -> !hlfir.expr<2x1xi32>
  return
}
// CHECK-LABEL:   func.func @transpose1(
// CHECK-SAME:                          %[[ARG0:.*]]: !hlfir.expr<1x2xi32>) {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[C2]], %[[C1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[EXPR:.*]] = hlfir.elemental %[[SHAPE]] : (!fir.shape<2>) -> !hlfir.expr<2x1xi32> {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             %[[ELEMENT:.*]] = hlfir.apply %[[ARG0]], %[[J]], %[[I]] : (!hlfir.expr<1x2xi32>, index, index) -> i32
// CHECK:             hlfir.yield_element %[[ELEMENT]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// box with unknown extent
func.func @transpose2(%arg0: !fir.box<!fir.array<?x2xi32>>) {
  %res = hlfir.transpose %arg0 : (!fir.box<!fir.array<?x2xi32>>) -> !hlfir.expr<2x?xi32>
  return
}
// CHECK-LABEL:   func.func @transpose2(
// CHECK-SAME:                          %[[ARG0:.*]]: !fir.box<!fir.array<?x2xi32>>) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<?x2xi32>>, index) -> (index, index, index)
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[C2]], %[[DIMS0]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[EXPR:.*]] = hlfir.elemental %[[SHAPE]] : (!fir.shape<2>) -> !hlfir.expr<2x?xi32> {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:             %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0_1]] : (!fir.box<!fir.array<?x2xi32>>, index) -> (index, index, index)
// CHECK:             %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:             %[[DIMS1_1:.*]]:3 = fir.box_dims %[[ARG0]], %[[C1_1]] : (!fir.box<!fir.array<?x2xi32>>, index) -> (index, index, index)
// CHECK:             %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:             %[[LOWER_BOUND0:.*]] = arith.subi %[[DIMS0]]#0, %[[C1_2]] : index
// CHECK:             %[[J_OFFSET:.*]] = arith.addi %[[J]], %[[LOWER_BOUND0]] : index
// CHECK:             %[[LOWER_BOUND1:.*]] = arith.subi %[[DIMS1_1]]#0, %[[C1_2]] : index
// CHECK:             %[[I_OFFSET:.*]] = arith.addi %[[I]], %[[LOWER_BOUND1]] : index
// CHECK:             %[[ELE_REF:.*]] = hlfir.designate %[[ARG0]] (%[[J_OFFSET]], %[[I_OFFSET]])  : (!fir.box<!fir.array<?x2xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[ELEMENT:.*]] = fir.load %[[ELE_REF]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[ELEMENT]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// expr with unknown extent
func.func @transpose3(%arg0: !hlfir.expr<?x2xi32>) {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<?x2xi32>) -> !hlfir.expr<2x?xi32>
  return
}
// CHECK-LABEL:   func.func @transpose3(
// CHECK-SAME:                          %[[ARG0:.*]]: !hlfir.expr<?x2xi32>) {
// CHECK:           %[[IN_SHAPE:.*]] = hlfir.shape_of %[[ARG0]] : (!hlfir.expr<?x2xi32>) -> !fir.shape<2>
// CHECK:           %[[EXTENT0:.*]] = hlfir.get_extent %[[IN_SHAPE]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[OUT_SHAPE:.*]] = fir.shape %[[C2]], %[[EXTENT0]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[EXPR:.*]] = hlfir.elemental %[[OUT_SHAPE]] : (!fir.shape<2>) -> !hlfir.expr<2x?xi32> {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             %[[ELEMENT:.*]] = hlfir.apply %[[ARG0]], %[[J]], %[[I]] : (!hlfir.expr<?x2xi32>, index, index) -> i32
// CHECK:             hlfir.yield_element %[[ELEMENT]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
