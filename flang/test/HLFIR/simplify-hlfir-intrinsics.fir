// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

// box with known extents
func.func @transpose0(%arg0: !fir.box<!fir.array<1x2xi32>>) -> !hlfir.expr<2x1xi32> {
  %res = hlfir.transpose %arg0 : (!fir.box<!fir.array<1x2xi32>>) -> !hlfir.expr<2x1xi32>
  return %res : !hlfir.expr<2x1xi32>
}
// CHECK-LABEL:   func.func @transpose0(
// CHECK-SAME:                          %[[ARG0:.*]]: !fir.box<!fir.array<1x2xi32>>) -> !hlfir.expr<2x1xi32> {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[C2]], %[[C1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[EXPR:.*]] = hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<2>) -> !hlfir.expr<2x1xi32> {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<1x2xi32>>, index) -> (index, index, index)
// CHECK:             %[[DIMS1:.*]]:3 = fir.box_dims %[[ARG0]], %[[C1]] : (!fir.box<!fir.array<1x2xi32>>, index) -> (index, index, index)
// CHECK:             %[[LOWER_BOUND0:.*]] = arith.subi %[[DIMS0]]#0, %[[C1]] : index
// CHECK:             %[[J_OFFSET:.*]] = arith.addi %[[J]], %[[LOWER_BOUND0]] : index
// CHECK:             %[[LOWER_BOUND1:.*]] = arith.subi %[[DIMS1]]#0, %[[C1]] : index
// CHECK:             %[[I_OFFSET:.*]] = arith.addi %[[I]], %[[LOWER_BOUND1]] : index
// CHECK:             %[[ELEMENT_REF:.*]] = hlfir.designate %[[ARG0]] (%[[J_OFFSET]], %[[I_OFFSET]])  : (!fir.box<!fir.array<1x2xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[ELEMENT:.*]] = fir.load %[[ELEMENT_REF]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[ELEMENT]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// expr with known extents
func.func @transpose1(%arg0: !hlfir.expr<1x2xi32>) -> !hlfir.expr<2x1xi32> {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<1x2xi32>) -> !hlfir.expr<2x1xi32>
  return %res : !hlfir.expr<2x1xi32>
}
// CHECK-LABEL:   func.func @transpose1(
// CHECK-SAME:                          %[[ARG0:.*]]: !hlfir.expr<1x2xi32>) -> !hlfir.expr<2x1xi32> {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[C2]], %[[C1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[EXPR:.*]] = hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<2>) -> !hlfir.expr<2x1xi32> {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             %[[ELEMENT:.*]] = hlfir.apply %[[ARG0]], %[[J]], %[[I]] : (!hlfir.expr<1x2xi32>, index, index) -> i32
// CHECK:             hlfir.yield_element %[[ELEMENT]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// box with unknown extent
func.func @transpose2(%arg0: !fir.box<!fir.array<?x2xi32>>) -> !hlfir.expr<2x?xi32> {
  %res = hlfir.transpose %arg0 : (!fir.box<!fir.array<?x2xi32>>) -> !hlfir.expr<2x?xi32>
  return %res : !hlfir.expr<2x?xi32>
}
// CHECK-LABEL:   func.func @transpose2(
// CHECK-SAME:                          %[[ARG0:.*]]: !fir.box<!fir.array<?x2xi32>>) -> !hlfir.expr<2x?xi32> {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<?x2xi32>>, index) -> (index, index, index)
// CHECK:           %[[SHAPE:.*]] = fir.shape %[[C2]], %[[DIMS0]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[EXPR:.*]] = hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<2>) -> !hlfir.expr<2x?xi32> {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<?x2xi32>>, index) -> (index, index, index)
// CHECK:             %[[DIMS1_1:.*]]:3 = fir.box_dims %[[ARG0]], %[[C1]] : (!fir.box<!fir.array<?x2xi32>>, index) -> (index, index, index)
// CHECK:             %[[LOWER_BOUND0:.*]] = arith.subi %[[DIMS0]]#0, %[[C1]] : index
// CHECK:             %[[J_OFFSET:.*]] = arith.addi %[[J]], %[[LOWER_BOUND0]] : index
// CHECK:             %[[LOWER_BOUND1:.*]] = arith.subi %[[DIMS1_1]]#0, %[[C1]] : index
// CHECK:             %[[I_OFFSET:.*]] = arith.addi %[[I]], %[[LOWER_BOUND1]] : index
// CHECK:             %[[ELE_REF:.*]] = hlfir.designate %[[ARG0]] (%[[J_OFFSET]], %[[I_OFFSET]])  : (!fir.box<!fir.array<?x2xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[ELEMENT:.*]] = fir.load %[[ELE_REF]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[ELEMENT]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// expr with unknown extent
func.func @transpose3(%arg0: !hlfir.expr<?x2xi32>) -> !hlfir.expr<2x?xi32> {
  %res = hlfir.transpose %arg0 : (!hlfir.expr<?x2xi32>) -> !hlfir.expr<2x?xi32>
  return %res : !hlfir.expr<2x?xi32>
}
// CHECK-LABEL:   func.func @transpose3(
// CHECK-SAME:                          %[[ARG0:.*]]: !hlfir.expr<?x2xi32>) -> !hlfir.expr<2x?xi32> {
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[IN_SHAPE:.*]] = hlfir.shape_of %[[ARG0]] : (!hlfir.expr<?x2xi32>) -> !fir.shape<2>
// CHECK:           %[[EXTENT0:.*]] = hlfir.get_extent %[[IN_SHAPE]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[OUT_SHAPE:.*]] = fir.shape %[[C2]], %[[EXTENT0]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[EXPR:.*]] = hlfir.elemental %[[OUT_SHAPE]] unordered : (!fir.shape<2>) -> !hlfir.expr<2x?xi32> {
// CHECK:           ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:             %[[ELEMENT:.*]] = hlfir.apply %[[ARG0]], %[[J]], %[[I]] : (!hlfir.expr<?x2xi32>, index, index) -> i32
// CHECK:             hlfir.yield_element %[[ELEMENT]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// expr with multiple uses
func.func @transpose4(%arg0: !hlfir.expr<2x2xf32>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) {
  %0 = hlfir.transpose %arg0 : (!hlfir.expr<2x2xf32>) -> !hlfir.expr<2x2xf32>
  %1 = hlfir.shape_of %0 : (!hlfir.expr<2x2xf32>) -> !fir.shape<2>
  %2 = hlfir.elemental %1 : (!fir.shape<2>) -> !hlfir.expr<2x2xf32> {
  ^bb0(%arg2: index, %arg3: index):
    %3 = hlfir.apply %0, %arg2, %arg3 : (!hlfir.expr<2x2xf32>, index, index) -> f32
    %4 = math.cos %3 fastmath<contract> : f32
    hlfir.yield_element %4 : f32
  }
  hlfir.assign %2 to %arg1 realloc : !hlfir.expr<2x2xf32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  hlfir.destroy %2 : !hlfir.expr<2x2xf32>
  hlfir.destroy %0 : !hlfir.expr<2x2xf32>
  return
}
// CHECK-LABEL: func.func @transpose4(
// CHECK-SAME:      %[[ARG0:.*]]: !hlfir.expr<2x2xf32>
// CHECK-SAME:      %[[ARG1:.*]]:
// CHECK:         %[[SHAPE0:.*]] = fir.shape
// CHECK:         %[[TRANSPOSE:.*]] = hlfir.elemental %[[SHAPE0]] unordered : (!fir.shape<2>) -> !hlfir.expr<2x2xf32> {
// CHECK:         ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:           %[[ELE:.*]] = hlfir.apply %[[ARG0]], %[[J]], %[[I]] : (!hlfir.expr<2x2xf32>, index, index) -> f32
// CHECK:           hlfir.yield_element %[[ELE]] : f32
// CHECK:         }
// CHECK:         %[[COS:.*]] = hlfir.elemental %[[SHAPE0]] : (!fir.shape<2>) -> !hlfir.expr<2x2xf32> {
// CHECK:         ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:           %[[ELE:.*]] = hlfir.apply %[[TRANSPOSE]], %[[I]], %[[J]] : (!hlfir.expr<2x2xf32>, index, index) -> f32
// CHECK:           %[[COS_ELE:.*]] = math.cos %[[ELE]] fastmath<contract> : f32
// CHECK:           hlfir.yield_element %[[COS_ELE]] : f32
// CHECK:         }
// CHECK:         hlfir.assign %[[COS]] to %[[ARG1]] realloc
// CHECK:         hlfir.destroy %[[COS]] : !hlfir.expr<2x2xf32>
// CHECK:         hlfir.destroy %[[TRANSPOSE]] : !hlfir.expr<2x2xf32>
// CHECK:         return
// CHECK:       }

// regression test
func.func @transpose5(%arg0: !fir.ref<tuple<!fir.box<!fir.array<2x2xf64>>, !fir.box<!fir.array<2x2xf64>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  %0 = fir.address_of(@_QFEb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf64>>>>
  %1:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEb"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf64>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf64>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf64>>>>)
  %c0_i32 = arith.constant 0 : i32
  %2 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<2x2xf64>>, !fir.box<!fir.array<2x2xf64>>>>, i32) -> !fir.ref<!fir.box<!fir.array<2x2xf64>>>
  %3 = fir.load %2 : !fir.ref<!fir.box<!fir.array<2x2xf64>>>
  %4 = fir.box_addr %3 : (!fir.box<!fir.array<2x2xf64>>) -> !fir.ref<!fir.array<2x2xf64>>
  %c0 = arith.constant 0 : index
  %5:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.array<2x2xf64>>, index) -> (index, index, index)
  %c1 = arith.constant 1 : index
  %6:3 = fir.box_dims %3, %c1 : (!fir.box<!fir.array<2x2xf64>>, index) -> (index, index, index)
  %7 = fir.shape %5#1, %6#1 : (index, index) -> !fir.shape<2>
  %8:2 = hlfir.declare %4(%7) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<2x2xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x2xf64>>, !fir.ref<!fir.array<2x2xf64>>)
  %c1_i32 = arith.constant 1 : i32
  %9 = fir.coordinate_of %arg0, %c1_i32 : (!fir.ref<tuple<!fir.box<!fir.array<2x2xf64>>, !fir.box<!fir.array<2x2xf64>>>>, i32) -> !fir.ref<!fir.box<!fir.array<2x2xf64>>>
  %10 = fir.load %9 : !fir.ref<!fir.box<!fir.array<2x2xf64>>>
  %11 = fir.box_addr %10 : (!fir.box<!fir.array<2x2xf64>>) -> !fir.ref<!fir.array<2x2xf64>>
  %c0_0 = arith.constant 0 : index
  %12:3 = fir.box_dims %10, %c0_0 : (!fir.box<!fir.array<2x2xf64>>, index) -> (index, index, index)
  %c1_1 = arith.constant 1 : index
  %13:3 = fir.box_dims %10, %c1_1 : (!fir.box<!fir.array<2x2xf64>>, index) -> (index, index, index)
  %14 = fir.shape %12#1, %13#1 : (index, index) -> !fir.shape<2>
  %15:2 = hlfir.declare %11(%14) {uniq_name = "_QFEc"} : (!fir.ref<!fir.array<2x2xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x2xf64>>, !fir.ref<!fir.array<2x2xf64>>)
  %16 = hlfir.transpose %8#0 : (!fir.ref<!fir.array<2x2xf64>>) -> !hlfir.expr<2x2xf64>
  %17 = hlfir.shape_of %16 : (!hlfir.expr<2x2xf64>) -> !fir.shape<2>
  %18 = hlfir.elemental %17 : (!fir.shape<2>) -> !hlfir.expr<?x?xf64> {
  ^bb0(%arg1: index, %arg2: index):
    %19 = hlfir.apply %16, %arg1, %arg2 : (!hlfir.expr<2x2xf64>, index, index) -> f64
    %20 = math.cos %19 fastmath<contract> : f64
    hlfir.yield_element %20 : f64
  }
  hlfir.assign %18 to %1#0 realloc : !hlfir.expr<?x?xf64>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf64>>>>
  hlfir.destroy %18 : !hlfir.expr<?x?xf64>
  hlfir.destroy %16 : !hlfir.expr<2x2xf64>
  return
}
// CHECK-LABEL: func.func @transpose5(
// ...
// CHECK:         %[[TRANSPOSE:.*]] = hlfir.elemental %[[SHAPE0:[A-Za-z._]*]]
// CHECK:         ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:           %[[ELE:.*]] = hlfir.designate %[[ARRAY:.*]] (%[[J]], %[[I]])
// CHECK:           %[[LOAD:.*]] = fir.load %[[ELE]]
// CHECK:           hlfir.yield_element %[[LOAD]]
// CHECK:         }
// CHECK:         %[[COS:.*]] = hlfir.elemental %[[SHAPE0]]
// ...
// CHECK:         hlfir.assign %[[COS]] to %{{.*}} realloc
// CHECK:         hlfir.destroy %[[COS]]
// CHECK:         hlfir.destroy %[[TRANSPOSE]]
// CHECK:         return
// CHECK:       }
