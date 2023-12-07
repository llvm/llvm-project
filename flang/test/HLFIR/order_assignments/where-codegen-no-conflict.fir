// Test code generation of hlfir.where, and hlfir.elsewhere when there
// is no need to create temporary storage.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @test_simple(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %cst = arith.constant 4.200000e+01 : f32
  %0:2 = hlfir.declare %arg1 {uniq_name = "mask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  hlfir.where {
    hlfir.yield %0#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      hlfir.yield %cst : f32
    } to {
      hlfir.yield %1#0 : !fir.box<!fir.array<?xf32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_simple(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 4.200000e+01 : f32
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "mask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_9:.*]] = %[[VAL_8]] to %[[VAL_6]]#1 step %[[VAL_8]] {
// CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_9]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_12]] {
// CHECK:               %[[VAL_13:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_2]] to %[[VAL_13]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


func.func @test_elsewhere(%arg0: !fir.ref<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>, %arg2: !fir.ref<!fir.array<100xf32>>, %arg3: !fir.ref<!fir.array<100x!fir.logical<4>>>, %arg4: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "mask2"}) {
  %c100 = arith.constant 100 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg3(%0) {uniq_name = "mask"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %2:2 = hlfir.declare %arg4(%0) {uniq_name = "mask2"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  %3:2 = hlfir.declare %arg0(%0) {uniq_name = "x"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %4:2 = hlfir.declare %arg1(%0) {uniq_name = "y"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  %5:2 = hlfir.declare %arg2(%0) {uniq_name = "z"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  hlfir.where {
    hlfir.yield %1#0 : !fir.ref<!fir.array<100x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      hlfir.yield %4#0 : !fir.ref<!fir.array<100xf32>>
    } to {
      hlfir.yield %3#0 : !fir.ref<!fir.array<100xf32>>
    }
    hlfir.elsewhere mask {
      hlfir.yield %2#0 : !fir.ref<!fir.array<100x!fir.logical<4>>>
    } do {
      hlfir.region_assign {
        hlfir.yield %3#0 : !fir.ref<!fir.array<100xf32>>
      } to {
        hlfir.yield %4#0 : !fir.ref<!fir.array<100xf32>>
      }
      hlfir.elsewhere do {
        hlfir.region_assign {
          hlfir.yield %4#0 : !fir.ref<!fir.array<100xf32>>
        } to {
          hlfir.yield %5#0 : !fir.ref<!fir.array<100xf32>>
        }
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_elsewhere(
// CHECK-SAME:                              %[[VAL_0:[^:]*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                              %[[VAL_1:[^:]*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                              %[[VAL_2:[^:]*]]: !fir.ref<!fir.array<100xf32>>,
// CHECK-SAME:                              %[[VAL_3:[^:]*]]: !fir.ref<!fir.array<100x!fir.logical<4>>>,
// CHECK-SAME:                              %[[VAL_4:[^:]*]]: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "mask2"}) {
// CHECK:           %[[VAL_5:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_6]]) {uniq_name = "mask"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_6]]) {uniq_name = "mask2"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_6]]) {uniq_name = "x"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_6]]) {uniq_name = "y"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_2]](%[[VAL_6]]) {uniq_name = "z"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_13:.*]] = %[[VAL_12]] to %[[VAL_5]] step %[[VAL_12]] {
// CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_16]] {
// CHECK:               %[[VAL_17:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_18:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_17]] to %[[VAL_18]] : !fir.ref<f32>, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_20:.*]] = %[[VAL_19]] to %[[VAL_5]] step %[[VAL_19]] {
// CHECK:             %[[VAL_21:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_23]] {
// CHECK:             } else {
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.logical<4>) -> i1
// CHECK:               fir.if %[[VAL_26]] {
// CHECK:                 %[[VAL_27:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
// CHECK:                 %[[VAL_28:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_27]] to %[[VAL_28]] : !fir.ref<f32>, !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_30:.*]] = %[[VAL_29]] to %[[VAL_5]] step %[[VAL_29]] {
// CHECK:             %[[VAL_31:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_30]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_32:.*]] = fir.load %[[VAL_31]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_33]] {
// CHECK:             } else {
// CHECK:               %[[VAL_34:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_30]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (!fir.logical<4>) -> i1
// CHECK:               fir.if %[[VAL_36]] {
// CHECK:               } else {
// CHECK:                 %[[VAL_37:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_30]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
// CHECK:                 %[[VAL_38:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_30]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_37]] to %[[VAL_38]] : !fir.ref<f32>, !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @expr_tree(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0:2 = hlfir.declare %arg2 {uniq_name = "mask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  hlfir.where {
    %3 = fir.shape %c10 : (index) -> !fir.shape<1>
    %4 = hlfir.designate %2#0 (%c10:%c1:%c-1)  shape %3 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
    %5 = hlfir.elemental %3 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
    ^bb0(%arg3: index):
      %9 = hlfir.designate %4 (%arg3)  : (!fir.box<!fir.array<10xf32>>, index) -> !fir.ref<f32>
      %10 = fir.load %9 : !fir.ref<f32>
      %11 = math.absf %10 fastmath<contract> : f32
      hlfir.yield_element %11 : f32
    }
    %6 = hlfir.elemental %3 : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
    ^bb0(%arg3: index):
      %9 = hlfir.apply %5, %arg3 : (!hlfir.expr<10xf32>, index) -> f32
      %10 = arith.cmpf ogt, %9, %cst : f32
      %11 = fir.convert %10 : (i1) -> !fir.logical<4>
      hlfir.yield_element %11 : !fir.logical<4>
    }
    %7 = hlfir.elemental %3 : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
    ^bb0(%arg3: index):
      %9 = hlfir.apply %6, %arg3 : (!hlfir.expr<10x!fir.logical<4>>, index) -> !fir.logical<4>
      %10 = hlfir.no_reassoc %9 : !fir.logical<4>
      hlfir.yield_element %10 : !fir.logical<4>
    }
    %8 = hlfir.elemental %3 : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
    ^bb0(%arg3: index):
      %9 = hlfir.apply %7, %arg3 : (!hlfir.expr<10x!fir.logical<4>>, index) -> !fir.logical<4>
      %10 = hlfir.designate %0#0 (%arg3)  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
      %11 = fir.load %10 : !fir.ref<!fir.logical<4>>
      %12 = fir.convert %9 : (!fir.logical<4>) -> i1
      %13 = fir.convert %11 : (!fir.logical<4>) -> i1
      %14 = arith.andi %12, %13 : i1
      %15 = fir.convert %14 : (i1) -> !fir.logical<4>
      hlfir.yield_element %15 : !fir.logical<4>
    }
    hlfir.yield %8 : !hlfir.expr<10x!fir.logical<4>> cleanup {
      hlfir.destroy %8 : !hlfir.expr<10x!fir.logical<4>>
      hlfir.destroy %7 : !hlfir.expr<10x!fir.logical<4>>
      hlfir.destroy %6 : !hlfir.expr<10x!fir.logical<4>>
      hlfir.destroy %5 : !hlfir.expr<10xf32>
    }
  } do {
    hlfir.region_assign {
      hlfir.yield %2#0 : !fir.box<!fir.array<?xf32>>
    } to {
      hlfir.yield %1#0 : !fir.box<!fir.array<?xf32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @expr_tree(
// CHECK-SAME:                         %[[VAL_0:[^:]*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:                         %[[VAL_1:[^:]*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:                         %[[VAL_2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant -1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "mask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "y"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_11:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_6]]:%[[VAL_5]]:%[[VAL_4]])  shape %[[VAL_10]] : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_13:.*]] = %[[VAL_12]] to %[[VAL_6]] step %[[VAL_12]] {
// CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_11]] (%[[VAL_13]])  : (!fir.box<!fir.array<10xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<f32>
// CHECK:             %[[VAL_16:.*]] = math.absf %[[VAL_15]] fastmath<contract> : f32
// CHECK:             %[[VAL_17:.*]] = arith.cmpf ogt, %[[VAL_16]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i1) -> !fir.logical<4>
// CHECK:             %[[VAL_19:.*]] = hlfir.no_reassoc %[[VAL_18]] : !fir.logical<4>
// CHECK:             %[[VAL_20:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_13]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_19]] : (!fir.logical<4>) -> i1
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (!fir.logical<4>) -> i1
// CHECK:             %[[VAL_24:.*]] = arith.andi %[[VAL_22]], %[[VAL_23]] : i1
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i1) -> !fir.logical<4>
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.logical<4>) -> i1
// CHECK:             fir.if %[[VAL_26]] {
// CHECK:               %[[VAL_27:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_13]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_28:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_13]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_27]] to %[[VAL_28]] : !fir.ref<f32>, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @inside_forall(%arg0: !fir.ref<!fir.array<10x20xf32>>, %arg1: !fir.ref<!fir.array<20xf32>>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c10_i32 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "x"} : (!fir.ref<!fir.array<10x20xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x20xf32>>, !fir.ref<!fir.array<10x20xf32>>)
  %2 = fir.shape %c20 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "y"} : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<20xf32>>, !fir.ref<!fir.array<20xf32>>)
  hlfir.forall lb {
    hlfir.yield %c1_i32 : i32
  } ub {
    hlfir.yield %c10_i32 : i32
  }  (%arg2: i32) {
    hlfir.where {
      %4 = hlfir.elemental %2 : (!fir.shape<1>) -> !hlfir.expr<20x!fir.logical<4>> {
      ^bb0(%arg3: index):
        %5 = hlfir.designate %3#0 (%arg3)  : (!fir.ref<!fir.array<20xf32>>, index) -> !fir.ref<f32>
        %6 = fir.load %5 : !fir.ref<f32>
        %7 = arith.cmpf ogt, %6, %cst : f32
        %8 = fir.convert %7 : (i1) -> !fir.logical<4>
        hlfir.yield_element %8 : !fir.logical<4>
      }
      hlfir.yield %4 : !hlfir.expr<20x!fir.logical<4>> cleanup {
        hlfir.destroy %4 : !hlfir.expr<20x!fir.logical<4>>
      }
    } do {
      hlfir.region_assign {
        hlfir.yield %3#0 : !fir.ref<!fir.array<20xf32>>
      } to {
        %4 = fir.convert %arg2 : (i32) -> i64
        %5 = hlfir.designate %1#0 (%4, %c1:%c20:%c1)  shape %2 : (!fir.ref<!fir.array<10x20xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<20xf32>>
        hlfir.yield %5 : !fir.box<!fir.array<20xf32>>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @inside_forall(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.array<10x20xf32>>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !fir.ref<!fir.array<20xf32>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_8]]) {uniq_name = "x"} : (!fir.ref<!fir.array<10x20xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x20xf32>>, !fir.ref<!fir.array<10x20xf32>>)
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_10]]) {uniq_name = "y"} : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<20xf32>>, !fir.ref<!fir.array<20xf32>>)
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_15:.*]] = %[[VAL_12]] to %[[VAL_13]] step %[[VAL_14]] {
// CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (index) -> i32
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
// CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_18]], %[[VAL_2]]:%[[VAL_7]]:%[[VAL_2]])  shape %[[VAL_10]] : (!fir.ref<!fir.array<10x20xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<20xf32>>
// CHECK:             fir.do_loop %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_7]] step %[[VAL_17]] {
// CHECK:               %[[VAL_21:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<20xf32>>, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<f32>
// CHECK:               %[[VAL_23:.*]] = arith.cmpf ogt, %[[VAL_22]], %[[VAL_3]] : f32
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i1) -> !fir.logical<4>
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.logical<4>) -> i1
// CHECK:               fir.if %[[VAL_25]] {
// CHECK:                 %[[VAL_26:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<20xf32>>, index) -> !fir.ref<f32>
// CHECK:                 %[[VAL_27:.*]] = hlfir.designate %[[VAL_19]] (%[[VAL_20]])  : (!fir.box<!fir.array<20xf32>>, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_26]] to %[[VAL_27]] : !fir.ref<f32>, !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
