// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s

func.func @test_total_expr(%arg0: !hlfir.expr<?x?x!fir.logical<4>>) -> i32 {
  %0 = hlfir.count %arg0 : (!hlfir.expr<?x?x!fir.logical<4>>) -> i32
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_total_expr(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x!fir.logical<4>>) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x!fir.logical<4>>) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_7:.*]] = fir.do_loop %[[VAL_8:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] unordered iter_args(%[[VAL_9:.*]] = %[[VAL_3]]) -> (i32) {
// CHECK:             %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_2]] to %[[VAL_5]] step %[[VAL_2]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_9]]) -> (i32) {
// CHECK:               %[[VAL_13:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]], %[[VAL_8]] : (!hlfir.expr<?x?x!fir.logical<4>>, index, index) -> !fir.logical<4>
// CHECK:               %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.logical<4>) -> i1
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : i32
// CHECK:               %[[VAL_16:.*]] = arith.select %[[VAL_14]], %[[VAL_15]], %[[VAL_12]] : i32
// CHECK:               fir.result %[[VAL_16]] : i32
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_7]] : i32
// CHECK:         }

func.func @test_partial_expr(%arg0: !hlfir.expr<?x?x?x!fir.logical<1>>) -> !hlfir.expr<?x?xi16> {
  %dim = arith.constant 2 : i32
  %0 = hlfir.count %arg0 dim %dim : (!hlfir.expr<?x?x?x!fir.logical<1>>, i32) -> !hlfir.expr<?x?xi16>
  return %0 : !hlfir.expr<?x?xi16>
}
// CHECK-LABEL:   func.func @test_partial_expr(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x?x!fir.logical<1>>) -> !hlfir.expr<?x?xi16> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i16
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i16
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x?x!fir.logical<1>>) -> !fir.shape<3>
// CHECK:           %[[VAL_5:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 0 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_6:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 1 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_7:.*]] = hlfir.get_extent %[[VAL_4]] {dim = 2 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_5]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_9:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi16> {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
// CHECK:             %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_2]] to %[[VAL_6]] step %[[VAL_2]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_3]]) -> (i16) {
// CHECK:               %[[VAL_15:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_10]], %[[VAL_13]], %[[VAL_11]] : (!hlfir.expr<?x?x?x!fir.logical<1>>, index, index, index) -> !fir.logical<1>
// CHECK:               %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.logical<1>) -> i1
// CHECK:               %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_1]] : i16
// CHECK:               %[[VAL_18:.*]] = arith.select %[[VAL_16]], %[[VAL_17]], %[[VAL_14]] : i16
// CHECK:               fir.result %[[VAL_18]] : i16
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_12]] : i16
// CHECK:           }
// CHECK:           return %[[VAL_9]] : !hlfir.expr<?x?xi16>
// CHECK:         }

func.func @test_total_var(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>>) -> i32 {
  %0 = hlfir.count %arg0 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> i32
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_total_var(
// CHECK-SAME:                              %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.do_loop %[[VAL_8:.*]] = %[[VAL_3]] to %[[VAL_6]]#1 step %[[VAL_3]] unordered iter_args(%[[VAL_9:.*]] = %[[VAL_2]]) -> (i32) {
// CHECK:             %[[VAL_10:.*]] = fir.do_loop %[[VAL_11:.*]] = %[[VAL_3]] to %[[VAL_5]]#1 step %[[VAL_3]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_9]]) -> (i32) {
// CHECK:               %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_15:.*]] = arith.subi %[[VAL_13]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_16:.*]] = arith.addi %[[VAL_11]], %[[VAL_15]] : index
// CHECK:               %[[VAL_17:.*]] = arith.subi %[[VAL_14]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_8]], %[[VAL_17]] : index
// CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_16]], %[[VAL_18]])  : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index, index) -> !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.logical<4>) -> i1
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : i32
// CHECK:               %[[VAL_23:.*]] = arith.select %[[VAL_21]], %[[VAL_22]], %[[VAL_12]] : i32
// CHECK:               fir.result %[[VAL_23]] : i32
// CHECK:             }
// CHECK:             fir.result %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_7]] : i32
// CHECK:         }

func.func @test_partial_var(%arg0: !fir.box<!fir.array<?x?x?x!fir.logical<2>>>) -> !hlfir.expr<?x?xi64> {
  %dim = arith.constant 2 : i32
  %0 = hlfir.count %arg0 dim %dim : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, i32) -> !hlfir.expr<?x?xi64>
  return %0 : !hlfir.expr<?x?xi64>
}
// CHECK-LABEL:   func.func @test_partial_var(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x?x!fir.logical<2>>>) -> !hlfir.expr<?x?xi64> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_6]]#1, %[[VAL_8]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi64> {
// CHECK:           ^bb0(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
// CHECK:             %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_4]] to %[[VAL_7]]#1 step %[[VAL_4]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_2]]) -> (i64) {
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_19:.*]] = arith.subi %[[VAL_16]]#0, %[[VAL_4]] : index
// CHECK:               %[[VAL_20:.*]] = arith.addi %[[VAL_11]], %[[VAL_19]] : index
// CHECK:               %[[VAL_21:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_4]] : index
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_14]], %[[VAL_21]] : index
// CHECK:               %[[VAL_23:.*]] = arith.subi %[[VAL_18]]#0, %[[VAL_4]] : index
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_12]], %[[VAL_23]] : index
// CHECK:               %[[VAL_25:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_20]], %[[VAL_22]], %[[VAL_24]])  : (!fir.box<!fir.array<?x?x?x!fir.logical<2>>>, index, index, index) -> !fir.ref<!fir.logical<2>>
// CHECK:               %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<!fir.logical<2>>
// CHECK:               %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (!fir.logical<2>) -> i1
// CHECK:               %[[VAL_28:.*]] = arith.addi %[[VAL_15]], %[[VAL_1]] : i64
// CHECK:               %[[VAL_29:.*]] = arith.select %[[VAL_27]], %[[VAL_28]], %[[VAL_15]] : i64
// CHECK:               fir.result %[[VAL_29]] : i64
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_13]] : i64
// CHECK:           }
// CHECK:           return %[[VAL_10]] : !hlfir.expr<?x?xi64>
// CHECK:         }
