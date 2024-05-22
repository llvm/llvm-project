// Test optimized bufferization for hlfir.assign with scalar RHS.
// RUN: fir-opt --opt-bufferization %s | FileCheck %s

func.func @_QPtest1() {
  %cst = arith.constant 0.000000e+00 : f32
  %c11 = arith.constant 11 : index
  %c13 = arith.constant 13 : index
  %0 = fir.alloca !fir.array<11x13xf32> {bindc_name = "x", uniq_name = "_QFtest1Ex"}
  %1 = fir.shape %c11, %c13 : (index, index) -> !fir.shape<2>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<11x13xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<11x13xf32>>, !fir.ref<!fir.array<11x13xf32>>)
  hlfir.assign %cst to %2#0 : f32, !fir.ref<!fir.array<11x13xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 11 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 13 : index
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.array<11x13xf32> {bindc_name = "x", uniq_name = "_QFtest1Ex"}
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_5]]) {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<11x13xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<11x13xf32>>, !fir.ref<!fir.array<11x13xf32>>)
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_0]] to %[[VAL_3]] step %[[VAL_0]] unordered {
// CHECK:             fir.do_loop %[[VAL_8:.*]] = %[[VAL_0]] to %[[VAL_2]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_8]], %[[VAL_7]])  : (!fir.ref<!fir.array<11x13xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_1]] to %[[VAL_9]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest2(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "x"}) {
  %c0_i32 = arith.constant 0 : i32
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  hlfir.assign %c0_i32 to %0#0 : i32, !fir.box<!fir.array<?x?xi32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_2]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_1]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_6]]#1 step %[[VAL_1]] unordered {
// CHECK:             fir.do_loop %[[VAL_8:.*]] = %[[VAL_1]] to %[[VAL_5]]#1 step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_9:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_8]], %[[VAL_7]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_3]] to %[[VAL_9]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest4(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "x"}) {
  %true = arith.constant true
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest4Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>)
  %1 = fir.convert %true : (i1) -> !fir.logical<4>
  %2 = fir.load %0#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
  hlfir.assign %1 to %2 : !fir.logical<4>, !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest4(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest4Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>)
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>
// CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
// CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_1]] to %[[VAL_7]]#1 step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]]#0, %[[VAL_1]] : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_10]] : index
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_6]] (%[[VAL_11]])  : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             hlfir.assign %[[VAL_5]] to %[[VAL_12]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest3(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"}) {
  %c0_i32 = arith.constant 0 : i32
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  hlfir.assign %c0_i32 to %0#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_6]]#1 step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]]#0, %[[VAL_1]] : index
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:             %[[VAL_11:.*]] = hlfir.designate %[[VAL_5]] (%[[VAL_10]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_3]] to %[[VAL_11]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest5(%arg0: !fir.ref<!fir.array<77x!fir.complex<4>>> {fir.bindc_name = "x"}) {
  %cst = arith.constant 0.000000e+00 : f32
  %c77 = arith.constant 77 : index
  %0 = fir.shape %c77 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.array<77x!fir.complex<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<77x!fir.complex<4>>>, !fir.ref<!fir.array<77x!fir.complex<4>>>)
  %2 = fir.undefined !fir.complex<4>
  %3 = fir.insert_value %2, %cst, [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
  %4 = fir.insert_value %3, %cst, [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
  hlfir.assign %4 to %1#0 : !fir.complex<4>, !fir.ref<!fir.array<77x!fir.complex<4>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest5(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<77x!fir.complex<4>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 77 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.array<77x!fir.complex<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<77x!fir.complex<4>>>, !fir.ref<!fir.array<77x!fir.complex<4>>>)
// CHECK:           %[[VAL_6:.*]] = fir.undefined !fir.complex<4>
// CHECK:           %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_2]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
// CHECK:           %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_2]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
// CHECK:           fir.do_loop %[[VAL_9:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<77x!fir.complex<4>>>, index) -> !fir.ref<!fir.complex<4>>
// CHECK:             hlfir.assign %[[VAL_8]] to %[[VAL_10]] : !fir.complex<4>, !fir.ref<!fir.complex<4>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
