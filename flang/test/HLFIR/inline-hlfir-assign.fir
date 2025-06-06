// Test inlining of hlfir.assign of arrays:
// RUN: fir-opt --inline-hlfir-assign %s | FileCheck %s

// The two assigns come from the following source forms:
//   y(:,:) = x(:,:)
//   y = x
func.func @_QPtest1(%arg0: !fir.ref<!fir.array<3x3xf32>> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %2 = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest1Ey"}
  %3:2 = hlfir.declare %2(%0) {uniq_name = "_QFtest1Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %4 = hlfir.designate %1#0 (%c1:%c3:%c1, %c1:%c3:%c1)  shape %0 : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  %5 = hlfir.designate %3#0 (%c1:%c3:%c1, %c1:%c3:%c1)  shape %0 : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  hlfir.assign %4 to %5 : !fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>
  hlfir.assign %1#0 to %3#0 : !fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest1(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<3x3xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest1Ey"}
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_3]]) {uniq_name = "_QFtest1Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
// CHECK:           %[[VAL_7:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_1]]:%[[VAL_2]]:%[[VAL_1]], %[[VAL_1]]:%[[VAL_2]]:%[[VAL_1]])  shape %[[VAL_3]] : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
// CHECK:           %[[VAL_8:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_1]]:%[[VAL_2]]:%[[VAL_1]], %[[VAL_1]]:%[[VAL_2]]:%[[VAL_1]])  shape %[[VAL_3]] : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
// CHECK:           fir.do_loop %[[VAL_9:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:             fir.do_loop %[[VAL_10:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_11:.*]] = hlfir.designate %[[VAL_7]] (%[[VAL_10]], %[[VAL_9]])  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<f32>
// CHECK:               %[[VAL_13:.*]] = hlfir.designate %[[VAL_8]] (%[[VAL_10]], %[[VAL_9]])  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_12]] to %[[VAL_13]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.do_loop %[[VAL_14:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:             fir.do_loop %[[VAL_15:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_16:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_15]], %[[VAL_14]])  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<f32>
// CHECK:               %[[VAL_18:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_15]], %[[VAL_14]])  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_17]] to %[[VAL_18]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


func.func @_QPtest2(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %1 = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest2Ey"}
  %2 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest2Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %4:3 = fir.box_dims %0#1, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  %5:3 = fir.box_dims %0#1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  %6 = arith.cmpi sgt, %4#1, %c0 : index
  %7 = arith.select %6, %4#1, %c0 : index
  %8 = arith.cmpi sgt, %5#1, %c0 : index
  %9 = arith.select %8, %5#1, %c0 : index
  %10 = fir.shape %7, %9 : (index, index) -> !fir.shape<2>
  %11 = hlfir.designate %0#0 (%c1:%4#1:%c1, %c1:%5#1:%c1)  shape %10 : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
  %12 = hlfir.designate %3#0 (%c1:%c3:%c1, %c1:%c3:%c1)  shape %2 : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  hlfir.assign %11 to %12 : !fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<3x3xf32>>
  hlfir.assign %0#0 to %3#0 : !fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<3x3xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest2Ey"}
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_3]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) {uniq_name = "_QFtest2Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
// CHECK:           %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_4]]#1, %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_4]]#1, %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_9]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_9]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_11]], %[[VAL_13]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_15:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_2]]:%[[VAL_8]]#1:%[[VAL_2]], %[[VAL_2]]:%[[VAL_9]]#1:%[[VAL_2]])  shape %[[VAL_14]] : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_2]]:%[[VAL_3]]:%[[VAL_2]], %[[VAL_2]]:%[[VAL_3]]:%[[VAL_2]])  shape %[[VAL_6]] : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
// CHECK:           fir.do_loop %[[VAL_17:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] unordered {
// CHECK:             fir.do_loop %[[VAL_18:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] unordered {
// CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_15]] (%[[VAL_18]], %[[VAL_17]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<f32>
// CHECK:               %[[VAL_21:.*]] = hlfir.designate %[[VAL_16]] (%[[VAL_18]], %[[VAL_17]])  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_20]] to %[[VAL_21]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.do_loop %[[VAL_22:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] unordered {
// CHECK:             fir.do_loop %[[VAL_23:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] unordered {
// CHECK:               %[[VAL_24:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_23]], %[[VAL_22]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_24]] : !fir.ref<f32>
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_23]], %[[VAL_22]])  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_25]] to %[[VAL_26]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


func.func @_QPtest3(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %1 = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest3Ey"}
  %2 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest3Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %4 = hlfir.designate %3#0 (%c1:%c3:%c1, %c1:%c3:%c1)  shape %2 : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  %5:3 = fir.box_dims %0#1, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  %6:3 = fir.box_dims %0#1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  %7 = arith.cmpi sgt, %5#1, %c0 : index
  %8 = arith.select %7, %5#1, %c0 : index
  %9 = arith.cmpi sgt, %6#1, %c0 : index
  %10 = arith.select %9, %6#1, %c0 : index
  %11 = fir.shape %8, %10 : (index, index) -> !fir.shape<2>
  %12 = hlfir.designate %0#0 (%c1:%5#1:%c1, %c1:%6#1:%c1)  shape %11 : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
  hlfir.assign %4 to %12 : !fir.ref<!fir.array<3x3xf32>>, !fir.box<!fir.array<?x?xf32>>
  hlfir.assign %3#0 to %0#0 : !fir.ref<!fir.array<3x3xf32>>, !fir.box<!fir.array<?x?xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest3Ey"}
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_3]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) {uniq_name = "_QFtest3Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
// CHECK:           %[[VAL_8:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_2]]:%[[VAL_3]]:%[[VAL_2]], %[[VAL_2]]:%[[VAL_3]]:%[[VAL_2]])  shape %[[VAL_6]] : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
// CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_4]]#1, %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_4]]#1, %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_9]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_9]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_10]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_10]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_12]], %[[VAL_14]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_16:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_2]]:%[[VAL_9]]#1:%[[VAL_2]], %[[VAL_2]]:%[[VAL_10]]#1:%[[VAL_2]])  shape %[[VAL_15]] : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:           fir.do_loop %[[VAL_17:.*]] = %[[VAL_2]] to %[[VAL_14]] step %[[VAL_2]] unordered {
// CHECK:             fir.do_loop %[[VAL_18:.*]] = %[[VAL_2]] to %[[VAL_12]] step %[[VAL_2]] unordered {
// CHECK:               %[[VAL_19:.*]] = hlfir.designate %[[VAL_8]] (%[[VAL_18]], %[[VAL_17]])  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<f32>
// CHECK:               %[[VAL_21:.*]] = hlfir.designate %[[VAL_16]] (%[[VAL_18]], %[[VAL_17]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_20]] to %[[VAL_21]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_23:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_24:.*]] = %[[VAL_2]] to %[[VAL_23]]#1 step %[[VAL_2]] unordered {
// CHECK:             fir.do_loop %[[VAL_25:.*]] = %[[VAL_2]] to %[[VAL_22]]#1 step %[[VAL_2]] unordered {
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_25]], %[[VAL_24]])  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<f32>
// CHECK:               %[[VAL_28:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_25]], %[[VAL_24]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_27]] to %[[VAL_28]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


// The LHS is a whole allocatable, so the assignment may imply
// allocation. This is not currently supported.
func.func @_QPtest4(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
  %c3 = arith.constant 3 : index
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest4Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
  %1 = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest4Ey"}
  %2 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest4Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  hlfir.assign %3#0 to %0#0 realloc : !fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest4(
// CHECK-NOT:       hlfir.assign
// CHECK:           hlfir.assign %{{.*}} to %{{.*}} realloc : !fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
// CHECK-NOT:       hlfir.assign


// LHS is a pointer, but RHS is a subroutine local,
// so they cannot alias.
func.func @_QPtest5(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
  %1 = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest5Ey"}
  %2 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest5Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %4 = hlfir.designate %3#0 (%c1:%c3:%c1, %c1:%c3:%c1)  shape %2 : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  %5 = fir.load %0#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  %6:3 = fir.box_dims %5, %c0 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %7:3 = fir.box_dims %5, %c1 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %8 = arith.addi %6#0, %6#1 : index
  %9 = arith.subi %8, %c1 : index
  %10 = arith.addi %7#0, %7#1 : index
  %11 = arith.subi %10, %c1 : index
  %12 = arith.subi %9, %6#0 : index
  %13 = arith.addi %12, %c1 : index
  %14 = arith.cmpi sgt, %13, %c0 : index
  %15 = arith.select %14, %13, %c0 : index
  %16 = arith.subi %11, %7#0 : index
  %17 = arith.addi %16, %c1 : index
  %18 = arith.cmpi sgt, %17, %c0 : index
  %19 = arith.select %18, %17, %c0 : index
  %20 = fir.shape %15, %19 : (index, index) -> !fir.shape<2>
  %21 = hlfir.designate %5 (%6#0:%9:%c1, %7#0:%11:%c1)  shape %20 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
  hlfir.assign %4 to %21 : !fir.ref<!fir.array<3x3xf32>>, !fir.box<!fir.array<?x?xf32>>
  %22 = fir.load %0#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  hlfir.assign %3#0 to %22 : !fir.ref<!fir.array<3x3xf32>>, !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest5(
// CHECK-NOT:       hlfir.assign
// CHECK:           hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK-NOT:       hlfir.assign
// CHECK:           hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK-NOT:       hlfir.assign


// RHS is a pointer, but LHS is a subroutine local,
// so they cannot alias.
func.func @_QPtest6(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
  %1 = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest6Ey"}
  %2 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest6Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %4 = fir.load %0#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  %5:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %6:3 = fir.box_dims %4, %c1 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %7 = arith.addi %5#0, %5#1 : index
  %8 = arith.subi %7, %c1 : index
  %9 = arith.addi %6#0, %6#1 : index
  %10 = arith.subi %9, %c1 : index
  %11 = arith.subi %8, %5#0 : index
  %12 = arith.addi %11, %c1 : index
  %13 = arith.cmpi sgt, %12, %c0 : index
  %14 = arith.select %13, %12, %c0 : index
  %15 = arith.subi %10, %6#0 : index
  %16 = arith.addi %15, %c1 : index
  %17 = arith.cmpi sgt, %16, %c0 : index
  %18 = arith.select %17, %16, %c0 : index
  %19 = fir.shape %14, %18 : (index, index) -> !fir.shape<2>
  %20 = hlfir.designate %4 (%5#0:%8:%c1, %6#0:%10:%c1)  shape %19 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
  %21 = hlfir.designate %3#0 (%c1:%c3:%c1, %c1:%c3:%c1)  shape %2 : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  hlfir.assign %20 to %21 : !fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<3x3xf32>>
  %22 = fir.load %0#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  hlfir.assign %22 to %3#0 : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.ref<!fir.array<3x3xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest6(
// CHECK-NOT:       hlfir.assign
// CHECK:           hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK-NOT:       hlfir.assign
// CHECK:           hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK-NOT:       hlfir.assign


// LHS and RHS do not alias, and the assignment cannot
// allocate/reallocate LHS, so we should be able to optimize.
func.func @_QPtest7(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest7Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
  %1 = fir.alloca !fir.array<3x3xf32> {bindc_name = "y", uniq_name = "_QFtest7Ey"}
  %2 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtest7Ey"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %4 = hlfir.designate %3#0 (%c1:%c3:%c1, %c1:%c3:%c1)  shape %2 : (!fir.ref<!fir.array<3x3xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x3xf32>>
  %5 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  %6:3 = fir.box_dims %5, %c0 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %7:3 = fir.box_dims %5, %c1 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %8 = arith.addi %6#0, %6#1 : index
  %9 = arith.subi %8, %c1 : index
  %10 = arith.addi %7#0, %7#1 : index
  %11 = arith.subi %10, %c1 : index
  %12 = arith.subi %9, %6#0 : index
  %13 = arith.addi %12, %c1 : index
  %14 = arith.cmpi sgt, %13, %c0 : index
  %15 = arith.select %14, %13, %c0 : index
  %16 = arith.subi %11, %7#0 : index
  %17 = arith.addi %16, %c1 : index
  %18 = arith.cmpi sgt, %17, %c0 : index
  %19 = arith.select %18, %17, %c0 : index
  %20 = fir.shape %15, %19 : (index, index) -> !fir.shape<2>
  %21 = hlfir.designate %5 (%6#0:%9:%c1, %7#0:%11:%c1)  shape %20 : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
  hlfir.assign %4 to %21 : !fir.ref<!fir.array<3x3xf32>>, !fir.box<!fir.array<?x?xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest7(
// CHECK-NOT:       hlfir.assign
// CHECK:           hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK-NOT:       hlfir.assign


// Test that VAR = EXPR assignment is inlined:
// subroutine test_expr_rhs(p1, p2)
//   logical, pointer :: p1(:), p2(:)
//   p1 = (p2)
// end subroutine test_expr_rhs
func.func @_QPtest_expr_rhs(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "p1"}, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "p2"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_expr_rhsEp1"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>)
  %2:2 = hlfir.declare %arg1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_expr_rhsEp2"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>)
  %3 = fir.load %2#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
  %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg2: index):
    %8 = arith.subi %4#0, %c1 : index
    %9 = arith.addi %arg2, %8 : index
    %10 = hlfir.designate %3 (%9)  : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> !fir.ref<!fir.logical<4>>
    %11 = fir.load %10 : !fir.ref<!fir.logical<4>>
    %12 = hlfir.no_reassoc %11 : !fir.logical<4>
    hlfir.yield_element %12 : !fir.logical<4>
  }
  %7 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
  hlfir.assign %6 to %7 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>
  hlfir.destroy %6 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK-LABEL:   func.func @_QPtest_expr_rhs(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "p1"},
// CHECK-SAME:                                %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "p2"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_4]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_expr_rhsEp1"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>)
// CHECK:           %[[VAL_10:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
// CHECK:           %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_17]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_19:.*]] = %[[VAL_2]] to %[[VAL_18]]#1 step %[[VAL_2]] unordered {
// CHECK:             %[[VAL_20:.*]] = hlfir.apply %[[VAL_10]], %[[VAL_19]] : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
// CHECK:             %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_17]], %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_22:.*]] = arith.subi %[[VAL_21]]#0, %[[VAL_2]] : index
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_19]], %[[VAL_22]] : index
// CHECK:             %[[VAL_24:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_23]])  : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             hlfir.assign %[[VAL_20]] to %[[VAL_24]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:           }
// CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<?x!fir.logical<4>>
// CHECK:           return
// CHECK:         }
