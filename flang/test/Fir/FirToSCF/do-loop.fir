// RUN: fir-opt %s --fir-to-scf | FileCheck %s

// CHECK-LABEL:   func.func @simple_loop(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>>) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : index
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_0]] : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_5]], %[[VAL_0]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_8]] {
// CHECK:             %[[VAL_10:.*]] = arith.muli %[[VAL_9]], %[[VAL_0]] : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_0]], %[[VAL_10]] : index
// CHECK:             %[[VAL_12:.*]] = fir.array_coor %[[ARG0]](%[[VAL_2]]) %[[VAL_11]] : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:             fir.store %[[VAL_3]] to %[[VAL_12]] : !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @simple_loop(%arg0: !fir.ref<!fir.array<100xi32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %c1_i32 = arith.constant 1 : i32
  fir.do_loop %arg1 = %c1 to %c100 step %c1 {
    %1 = fir.array_coor %arg0(%0) %arg1 : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %c1_i32 to %1 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL:   func.func @loop_with_negtive_step(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>>) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant -1 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : index
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_2]] : index
// CHECK:           %[[VAL_7:.*]] = arith.divsi %[[VAL_6]], %[[VAL_2]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_9]] {
// CHECK:             %[[VAL_11:.*]] = arith.muli %[[VAL_10]], %[[VAL_2]] : index
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_0]], %[[VAL_11]] : index
// CHECK:             %[[VAL_13:.*]] = fir.array_coor %[[ARG0]](%[[VAL_3]]) %[[VAL_12]] : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:             fir.store %[[VAL_4]] to %[[VAL_13]] : !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @loop_with_negtive_step(%arg0: !fir.ref<!fir.array<100xi32>>) {
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %c1_i32 = arith.constant 1 : i32
  fir.do_loop %arg1 = %c100 to %c1 step %c-1 {
    %1 = fir.array_coor %arg0(%0) %arg1 : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %c1_i32 to %1 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL:   func.func @loop_with_results(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32>) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_0]] : index
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_0]] : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_5]], %[[VAL_0]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_9:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_8]] iter_args(%[[VAL_11:.*]] = %[[VAL_1]]) -> (i32) {
// CHECK:             %[[VAL_12:.*]] = arith.muli %[[VAL_10]], %[[VAL_0]] : index
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_0]], %[[VAL_12]] : index
// CHECK:             %[[VAL_14:.*]] = fir.array_coor %[[ARG0]](%[[VAL_3]]) %[[VAL_13]] : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_11]], %[[VAL_15]] : i32
// CHECK:             scf.yield %[[VAL_16]] : i32
// CHECK:           }
// CHECK:           fir.store %[[VAL_9]] to %[[ARG1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @loop_with_results(%arg0: !fir.ref<!fir.array<100xi32>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.do_loop %arg2 = %c1 to %c100 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
    %2 = fir.array_coor %arg0(%0) %arg2 : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %3 = fir.load %2 : !fir.ref<i32>
    %4 = arith.addi %arg3, %3 : i32
    fir.result %4 : i32
  }
  fir.store %1 to %arg1 : !fir.ref<i32>
  return
}

// CHECK-LABEL:   func.func @loop_with_final_value(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32>) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_3:.*]] = fir.alloca index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_2]], %[[VAL_0]] : index
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_0]] : index
// CHECK:           %[[VAL_7:.*]] = arith.divsi %[[VAL_6]], %[[VAL_0]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]]:2 = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_9]] iter_args(%[[VAL_12:.*]] = %[[VAL_0]], %[[VAL_13:.*]] = %[[VAL_1]]) -> (index, i32) {
// CHECK:             %[[VAL_14:.*]] = arith.muli %[[VAL_11]], %[[VAL_0]] : index
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_0]], %[[VAL_14]] : index
// CHECK:             %[[VAL_16:.*]] = fir.array_coor %[[ARG0]](%[[VAL_4]]) %[[VAL_15]] : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_0]] overflow<nsw> : index
// CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_13]], %[[VAL_17]] overflow<nsw> : i32
// CHECK:             scf.yield %[[VAL_18]], %[[VAL_19]] : index, i32
// CHECK:           }
// CHECK:           fir.store %[[VAL_20:.*]]#0 to %[[VAL_3]] : !fir.ref<index>
// CHECK:           fir.store %[[VAL_20]]#1 to %[[ARG1]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
func.func @loop_with_final_value(%arg0: !fir.ref<!fir.array<100xi32>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.alloca index
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = fir.do_loop %arg2 = %c1 to %c100 step %c1 iter_args(%arg3 = %c0_i32) -> (index, i32) {
    %3 = fir.array_coor %arg0(%1) %arg2 : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %4 = fir.load %3 : !fir.ref<i32>
    %5 = arith.addi %arg2, %c1 overflow<nsw> : index
    %6 = arith.addi %arg3, %4 overflow<nsw> : i32
    fir.result %5, %6 : index, i32
  }
  fir.store %2#0 to %0 : !fir.ref<index>
  fir.store %2#1 to %arg1 : !fir.ref<i32>
  return
}

// CHECK-LABEL:   func.func @loop_with_attribute(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>>,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32>) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_2]], %[[VAL_0]] : index
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_0]] : index
// CHECK:           %[[VAL_7:.*]] = arith.divsi %[[VAL_6]], %[[VAL_0]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_9]] {
// CHECK:             %[[VAL_11:.*]] = arith.muli %[[VAL_10]], %[[VAL_0]] : index
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_0]], %[[VAL_11]] : index
// CHECK:             %[[VAL_13:.*]] = fir.array_coor %[[ARG0]](%[[VAL_4]]) %[[VAL_12]] : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
// CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_14]] : i32
// CHECK:             fir.store %[[VAL_16]] to %[[VAL_3]] : !fir.ref<i32>
// CHECK:           } {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, reduceAttrs = [#fir.reduce_attr<add>]}
// CHECK:           return
// CHECK:         }
func.func @loop_with_attribute(%arg0: !fir.ref<!fir.array<100xi32>>, %arg1: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.alloca i32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  fir.do_loop %arg2 = %c1 to %c100 step %c1 reduce(#fir.reduce_attr<add> -> %0 : !fir.ref<i32>) {
    %2 = fir.array_coor %arg0(%1) %arg2 : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %3 = fir.load %2 : !fir.ref<i32>
    %4 = fir.load %0 : !fir.ref<i32>
    %5 = arith.addi %4, %3 : i32
    fir.store %5 to %0 : !fir.ref<i32>
    fir.result
  }
  return
}

// CHECK-LABEL:   func.func @nested_loop(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100x100xi32>>) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_0]] : index
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_0]] : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_5]], %[[VAL_0]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_8]] {
// CHECK:             %[[VAL_10:.*]] = arith.muli %[[VAL_9]], %[[VAL_0]] : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_0]], %[[VAL_10]] : index
// CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_2]], %[[VAL_0]] : index
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_0]] : index
// CHECK:             %[[VAL_14:.*]] = arith.divsi %[[VAL_13]], %[[VAL_0]] : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_17:.*]] = %[[VAL_15]] to %[[VAL_14]] step %[[VAL_16]] {
// CHECK:               %[[VAL_18:.*]] = arith.muli %[[VAL_17]], %[[VAL_0]] : index
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_0]], %[[VAL_18]] : index
// CHECK:               %[[VAL_20:.*]] = fir.array_coor %[[ARG0]](%[[VAL_3]]) %[[VAL_19]], %[[VAL_11]] : (!fir.ref<!fir.array<100x100xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
// CHECK:               fir.store %[[VAL_1]] to %[[VAL_20]] : !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @nested_loop(%arg0: !fir.ref<!fir.array<100x100xi32>>) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100, %c100 : (index, index) -> !fir.shape<2>
  fir.do_loop %arg1 = %c1 to %c100 step %c1 {
    fir.do_loop %arg2 = %c1 to %c100 step %c1 {
      %1 = fir.array_coor %arg0(%0) %arg2, %arg1 : (!fir.ref<!fir.array<100x100xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      fir.store %c1_i32 to %1 : !fir.ref<i32>
    }
  }
  return
}
