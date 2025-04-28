// Test code generation of hlfir.region_assign where the left-hand
// side terminator is an hlfir.elemental_addr (Fortran assignments to
// vector subscripted designators).
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @simple(%arg0: !fir.ref<!fir.array<100xf32>> , %arg1: !fir.ref<!fir.array<10xi64>> , %arg2: !fir.ref<!fir.array<10xf32>> ) {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFsimpleEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
  %4:2 = hlfir.declare %arg2(%2) {uniq_name = "z"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  hlfir.region_assign {
    hlfir.yield %4#0 : !fir.ref<!fir.array<10xf32>>
  } to {
    hlfir.elemental_addr %2 : !fir.shape<1> {
    ^bb0(%arg3: index):
      %5 = hlfir.designate %3#0 (%arg3)  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
      %6 = fir.load %5 : !fir.ref<i64>
      %7 = hlfir.designate %1#0 (%6)  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
      hlfir.yield %7 : !fir.ref<f32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @simple(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi64>>,
// CHECK-SAME:                      %[[VAL_2:.*]]: !fir.ref<!fir.array<10xf32>>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "_QFsimpleEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_7]]) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_7]]) {uniq_name = "z"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_11:.*]] = %[[VAL_10]] to %[[VAL_3]] step %[[VAL_10]] {
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
// CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i64>
// CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
// CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_15]] to %[[VAL_14]] : !fir.ref<f32>, !fir.ref<f32>
// CHECK:           }

func.func @forall_vector_lhs(%arg0: !fir.ref<!fir.array<100x20xf32>> , %arg1: !fir.ref<!fir.array<10xi64>> ) {
  %c20_i32 = arith.constant 20 : i32
  %c1_i32 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c100, %c20 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "x"} : (!fir.ref<!fir.array<100x20xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x20xf32>>, !fir.ref<!fir.array<100x20xf32>>)
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
  hlfir.forall lb {
    hlfir.yield %c1_i32 : i32
  } ub {
    hlfir.yield %c20_i32 : i32
  }  (%arg2: i32) {
    hlfir.region_assign {
      %4 = hlfir.elemental %2 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
      ^bb0(%arg3: index):
        %5 = hlfir.designate %3#0 (%arg3)  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
        %6 = fir.load %5 : !fir.ref<i64>
        %7 = fir.convert %6 : (i64) -> f32
        hlfir.yield_element %7 : f32
      }
      hlfir.yield %4 : !hlfir.expr<10xf32> cleanup {
        hlfir.destroy %4 : !hlfir.expr<10xf32>
      }
    } to {
      %4 = fir.convert %arg2 : (i32) -> i64
      hlfir.elemental_addr %2 : !fir.shape<1> {
      ^bb0(%arg3: index):
        %5 = hlfir.designate %3#0 (%arg3)  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
        %6 = fir.load %5 : !fir.ref<i64>
        %7 = hlfir.designate %1#0 (%6, %4)  : (!fir.ref<!fir.array<100x20xf32>>, i64, i64) -> !fir.ref<f32>
        hlfir.yield %7 : !fir.ref<f32>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @forall_vector_lhs(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.ref<!fir.array<100x20xf32>>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi64>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 20 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_5]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_7]]) {uniq_name = "x"} : (!fir.ref<!fir.array<100x20xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x20xf32>>, !fir.ref<!fir.array<100x20xf32>>)
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_9]]) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_14:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_13]] {
// CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
// CHECK:             %[[VAL_16:.*]] = hlfir.elemental %[[VAL_9]] : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
// CHECK:             ^bb0(%[[VAL_17:.*]]: index):
// CHECK:               %[[VAL_18:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
// CHECK:               %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<i64>
// CHECK:               %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> f32
// CHECK:               hlfir.yield_element %[[VAL_20]] : f32
// CHECK:             }
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:             fir.do_loop %[[VAL_23:.*]] = %[[VAL_22]] to %[[VAL_4]] step %[[VAL_22]] {
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_23]])  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
// CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<i64>
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_25]], %[[VAL_21]])  : (!fir.ref<!fir.array<100x20xf32>>, i64, i64) -> !fir.ref<f32>
// CHECK:               %[[VAL_27:.*]] = hlfir.apply %[[VAL_28:.*]], %[[VAL_23]] : (!hlfir.expr<10xf32>, index) -> f32
// CHECK:               hlfir.assign %[[VAL_27]] to %[[VAL_26]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:             hlfir.destroy %[[VAL_16]] : !hlfir.expr<10xf32>
// CHECK:           }

func.func @where_vector_subscripts(%arg0: !fir.ref<!fir.array<10x!fir.logical<4>>> , %arg1: !fir.ref<!fir.array<100xf32>> , %arg2: !fir.ref<!fir.array<10xi64>> ) {
  %cst = arith.constant 0.000000e+00 : f32
  %c100 = arith.constant 100 : index
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "l"} : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.ref<!fir.array<10x!fir.logical<4>>>)
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "x"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4:2 = hlfir.declare %arg2(%0) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
  hlfir.where {
    hlfir.yield %1#0 : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      hlfir.yield %cst : f32
    } to {
      hlfir.elemental_addr %0 : !fir.shape<1> {
      ^bb0(%arg3: index):
        %5 = hlfir.designate %4#0 (%arg3)  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
        %6 = fir.load %5 : !fir.ref<i64>
        %7 = hlfir.designate %3#0 (%6)  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
        hlfir.yield %7 : !fir.ref<f32>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @where_vector_subscripts(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                                       %[[VAL_2:.*]]: !fir.ref<!fir.array<10xi64>>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_6]]) {uniq_name = "l"} : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.ref<!fir.array<10x!fir.logical<4>>>)
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_8]]) {uniq_name = "x"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_6]]) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_12:.*]] = %[[VAL_11]] to %[[VAL_5]] step %[[VAL_11]] {
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_12]])  : (!fir.ref<!fir.array<10x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_15]] {
// CHECK:               %[[VAL_16:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_12]])  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
// CHECK:               %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i64>
// CHECK:               %[[VAL_18:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_3]] to %[[VAL_18]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }

func.func @unordered(%arg0: !fir.ref<!fir.array<100xf32>> , %arg1: !fir.ref<!fir.array<10xi64>> , %arg2: !fir.ref<!fir.array<10xf32>> ) {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFsimpleEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
  %4:2 = hlfir.declare %arg2(%2) {uniq_name = "z"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  hlfir.region_assign {
    hlfir.yield %4#0 : !fir.ref<!fir.array<10xf32>>
  } to {
    hlfir.elemental_addr %2 unordered : !fir.shape<1> {
    ^bb0(%arg3: index):
      %5 = hlfir.designate %3#0 (%arg3)  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
      %6 = fir.load %5 : !fir.ref<i64>
      %7 = hlfir.designate %1#0 (%6)  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
      hlfir.yield %7 : !fir.ref<f32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @unordered(
// CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:      %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi64>>,
// CHECK-SAME:      %[[VAL_2:.*]]: !fir.ref<!fir.array<10xf32>>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) {uniq_name = "_QFsimpleEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_7]]) {uniq_name = "y"} : (!fir.ref<!fir.array<10xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_7]]) {uniq_name = "z"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_11:.*]] = %[[VAL_10]] to %[[VAL_3]] step %[[VAL_10]] unordered {
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
// CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i64>
// CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
// CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_15]] to %[[VAL_14]] : !fir.ref<f32>, !fir.ref<f32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
