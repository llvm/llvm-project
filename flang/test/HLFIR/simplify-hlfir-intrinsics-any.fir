// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s

func.func @test_total_expr(%arg0: !hlfir.expr<?x?x!fir.logical<4>>) -> !fir.logical<4> {
  %0 = hlfir.any %arg0 : (!hlfir.expr<?x?x!fir.logical<4>>) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// CHECK-LABEL:   func.func @test_total_expr(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x!fir.logical<4>>) -> !fir.logical<4> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x!fir.logical<4>>) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 1 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_6:.*]] = fir.do_loop %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_5]] step %[[VAL_1]] unordered iter_args(%[[VAL_8:.*]] = %[[VAL_2]]) -> (i1) {
// CHECK:             %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_1]] unordered iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (i1) {
// CHECK:               %[[VAL_12:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_10]], %[[VAL_7]] : (!hlfir.expr<?x?x!fir.logical<4>>, index, index) -> !fir.logical<4>
// CHECK:               %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (!fir.logical<4>) -> i1
// CHECK:               %[[VAL_14:.*]] = arith.ori %[[VAL_13]], %[[VAL_11]] : i1
// CHECK:               fir.result %[[VAL_14]] : i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_9]] : i1
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_6]] : (i1) -> !fir.logical<4>
// CHECK:           return %[[VAL_15]] : !fir.logical<4>
// CHECK:         }

func.func @test_partial_expr(%arg0: !hlfir.expr<?x?x?x!fir.logical<1>>) -> !hlfir.expr<?x?x!fir.logical<1>> {
  %dim = arith.constant 2 : i32
  %0 = hlfir.any %arg0 dim %dim : (!hlfir.expr<?x?x?x!fir.logical<1>>, i32) -> !hlfir.expr<?x?x!fir.logical<1>>
  return %0 : !hlfir.expr<?x?x!fir.logical<1>>
}
// CHECK-LABEL:   func.func @test_partial_expr(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x?x!fir.logical<1>>) -> !hlfir.expr<?x?x!fir.logical<1>> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x?x!fir.logical<1>>) -> !fir.shape<3>
// CHECK:           %[[VAL_4:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 0 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 1 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_3]] {dim = 2 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_4]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_7]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<1>> {
// CHECK:           ^bb0(%[[VAL_9:.*]]: index, %[[VAL_10:.*]]: index):
// CHECK:             %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_1]] to %[[VAL_5]] step %[[VAL_1]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_2]]) -> (i1) {
// CHECK:               %[[VAL_14:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_9]], %[[VAL_12]], %[[VAL_10]] : (!hlfir.expr<?x?x?x!fir.logical<1>>, index, index, index) -> !fir.logical<1>
// CHECK:               %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<1>) -> i1
// CHECK:               %[[VAL_16:.*]] = arith.ori %[[VAL_15]], %[[VAL_13]] : i1
// CHECK:               fir.result %[[VAL_16]] : i1
// CHECK:             }
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_11]] : (i1) -> !fir.logical<1>
// CHECK:             hlfir.yield_element %[[VAL_17]] : !fir.logical<1>
// CHECK:           }
// CHECK:           return %[[VAL_8]] : !hlfir.expr<?x?x!fir.logical<1>>
// CHECK:         }

func.func @test_total_var(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.logical<4> {
  %0 = hlfir.any %arg0 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// CHECK-LABEL:   func.func @test_total_var(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.logical<4> {
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]] = fir.do_loop %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_5]]#1 step %[[VAL_2]] unordered iter_args(%[[VAL_8:.*]] = %[[VAL_1]]) -> (i1) {
// CHECK:             %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_2]] to %[[VAL_4]]#1 step %[[VAL_2]] unordered iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (i1) {
// CHECK:               %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]] = arith.subi %[[VAL_12]]#0, %[[VAL_2]] : index
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_10]], %[[VAL_14]] : index
// CHECK:               %[[VAL_16:.*]] = arith.subi %[[VAL_13]]#0, %[[VAL_2]] : index
// CHECK:               %[[VAL_17:.*]] = arith.addi %[[VAL_7]], %[[VAL_16]] : index
// CHECK:               %[[VAL_18:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_15]], %[[VAL_17]])  : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index, index) -> !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.logical<4>) -> i1
// CHECK:               %[[VAL_21:.*]] = arith.ori %[[VAL_20]], %[[VAL_11]] : i1
// CHECK:               fir.result %[[VAL_21]] : i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_9]] : i1
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_6]] : (i1) -> !fir.logical<4>
// CHECK:           return %[[VAL_22]] : !fir.logical<4>
// CHECK:         }

func.func @test_partial_var(%arg0: !fir.box<!fir.array<?x?x?x!fir.logical<2>>>) -> !hlfir.expr<?x?x!fir.logical<2>> {
  %dim = arith.constant 2 : i32
  %0 = hlfir.any %arg0 dim %dim : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, i32) -> !hlfir.expr<?x?x!fir.logical<2>>
  return %0 : !hlfir.expr<?x?x!fir.logical<2>>
}
// CHECK-LABEL:   func.func @test_partial_var(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x?x!fir.logical<2>>>) -> !hlfir.expr<?x?x!fir.logical<2>> {
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_5]]#1, %[[VAL_7]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_9:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<2>> {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
// CHECK:             %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_3]] to %[[VAL_6]]#1 step %[[VAL_3]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_1]]) -> (i1) {
// CHECK:               %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]] = arith.subi %[[VAL_15]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_10]], %[[VAL_18]] : index
// CHECK:               %[[VAL_20:.*]] = arith.subi %[[VAL_16]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_21:.*]] = arith.addi %[[VAL_13]], %[[VAL_20]] : index
// CHECK:               %[[VAL_22:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_11]], %[[VAL_22]] : index
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_19]], %[[VAL_21]], %[[VAL_23]])  : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index, index, index) -> !fir.ref<!fir.logical<2>>
// CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<!fir.logical<2>>
// CHECK:               %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.logical<2>) -> i1
// CHECK:               %[[VAL_27:.*]] = arith.ori %[[VAL_26]], %[[VAL_14]] : i1
// CHECK:               fir.result %[[VAL_27]] : i1
// CHECK:             }
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_12]] : (i1) -> !fir.logical<2>
// CHECK:             hlfir.yield_element %[[VAL_28]] : !fir.logical<2>
// CHECK:           }
// CHECK:           return %[[VAL_9]] : !hlfir.expr<?x?x!fir.logical<2>>
// CHECK:         }
