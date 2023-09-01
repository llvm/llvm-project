// Test optimized bufferization for hlfir.assign of array
// slices, e.g.:
//   x(2:7999,1:120,new) = (x(2:7999,1:120,old))
// We can expand hlfir.assign if the slices are either identical
// or completely disjoint. In case they are identical, we still
// need to make sure that the one-based indices are used
// uniformly for both LHS and RHS.
// RUN: fir-opt --opt-bufferization %s | FileCheck %s

func.func @_QPtest1(%arg0: !fir.ref<!fir.array<8000x120x3xf32>> {fir.bindc_name = "x"}) {
  %c7998 = arith.constant 7998 : index
  %c1 = arith.constant 1 : index
  %c7999 = arith.constant 7999 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c120 = arith.constant 120 : index
  %c8000 = arith.constant 8000 : index
  %0 = fir.alloca i32 {bindc_name = "new", uniq_name = "_QFtest1Enew"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFtest1Enew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "old", uniq_name = "_QFtest1Eold"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest1Eold"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.shape %c8000, %c120, %c3 : (index, index, index) -> !fir.shape<3>
  %5:2 = hlfir.declare %arg0(%4) {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<8000x120x3xf32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<8000x120x3xf32>>, !fir.ref<!fir.array<8000x120x3xf32>>)
  %6 = fir.load %3#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> i64
  %8 = fir.shape %c7998, %c120 : (index, index) -> !fir.shape<2>
  %9 = hlfir.designate %5#0 (%c2:%c7999:%c1, %c1:%c120:%c1, %7)  shape %8 : (!fir.ref<!fir.array<8000x120x3xf32>>, index, index, index, index, index, index, i64, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  %10 = hlfir.elemental %8 unordered : (!fir.shape<2>) -> !hlfir.expr<7998x120xf32> {
  ^bb0(%arg1: index, %arg2: index):
    %14 = hlfir.designate %9 (%arg1, %arg2)  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
    %15 = fir.load %14 : !fir.ref<f32>
    %16 = hlfir.no_reassoc %15 : f32
    hlfir.yield_element %16 : f32
  }
  %11 = fir.load %1#0 : !fir.ref<i32>
  %12 = fir.convert %11 : (i32) -> i64
  %13 = hlfir.designate %5#0 (%c2:%c7999:%c1, %c1:%c120:%c1, %12)  shape %8 : (!fir.ref<!fir.array<8000x120x3xf32>>, index, index, index, index, index, index, i64, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  hlfir.assign %10 to %13 : !hlfir.expr<7998x120xf32>, !fir.box<!fir.array<7998x120xf32>>
  hlfir.destroy %10 : !hlfir.expr<7998x120xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest1(
// CHECK:           fir.do_loop %[[VAL_21:.*]] =
// CHECK:             fir.do_loop %[[VAL_22:.*]] =
// CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_17:.*]] (%[[VAL_22]], %[[VAL_21]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<f32>
// CHECK:               %[[VAL_25:.*]] = hlfir.no_reassoc %[[VAL_24]] : f32
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_20:.*]] (%[[VAL_22]], %[[VAL_21]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_25]] to %[[VAL_26]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }

func.func @_QPtest2(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>> {fir.bindc_name = "x"}) {
  %c120 = arith.constant 120 : index
  %c7998 = arith.constant 7998 : index
  %c1 = arith.constant 1 : index
  %c7999 = arith.constant 7999 : index
  %c2 = arith.constant 2 : index
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest2Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>)
  %1 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>
  %2 = fir.shape %c7998, %c120 : (index, index) -> !fir.shape<2>
  %3 = hlfir.designate %1 (%c2:%c7999:%c1, %c1:%c120:%c1, %c2)  shape %2 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  %4 = hlfir.elemental %2 unordered : (!fir.shape<2>) -> !hlfir.expr<7998x120xf32> {
  ^bb0(%arg1: index, %arg2: index):
    %6 = hlfir.designate %3 (%arg1, %arg2)  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
    %7 = fir.load %6 : !fir.ref<f32>
    %8 = hlfir.no_reassoc %7 : f32
    hlfir.yield_element %8 : f32
  }
  %5 = hlfir.designate %1 (%c2:%c7999:%c1, %c1:%c120:%c1, %c1)  shape %2 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  hlfir.assign %4 to %5 : !hlfir.expr<7998x120xf32>, !fir.box<!fir.array<7998x120xf32>>
  hlfir.destroy %4 : !hlfir.expr<7998x120xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK:           fir.do_loop %[[VAL_11:.*]] =
// CHECK:             fir.do_loop %[[VAL_12:.*]] =
// CHECK:               %[[VAL_13:.*]] = hlfir.designate %[[VAL_9:.*]] (%[[VAL_12]], %[[VAL_11]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<f32>
// CHECK:               %[[VAL_15:.*]] = hlfir.no_reassoc %[[VAL_14]] : f32
// CHECK:               %[[VAL_16:.*]] = hlfir.designate %[[VAL_10:.*]] (%[[VAL_12]], %[[VAL_11]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_15]] to %[[VAL_16]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }

func.func @_QPtest3(%arg0: !fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>> {fir.bindc_name = "x"}) {
  %c7998 = arith.constant 7998 : index
  %c7999 = arith.constant 7999 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c120 = arith.constant 120 : index
  %c8000 = arith.constant 8000 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = fir.alloca i32 {bindc_name = "new", uniq_name = "_QFtest3Enew"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFtest3Enew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "old", uniq_name = "_QFtest3Eold"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest3Eold"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg0(%4) {uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>>, !fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>>)
  %6 = hlfir.designate %5#0 (%c1)  : (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>
  %7 = fir.shape %c8000, %c120, %c3 : (index, index, index) -> !fir.shape<3>
  %8 = fir.load %3#0 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> i64
  %10 = fir.shape %c7998, %c120 : (index, index) -> !fir.shape<2>
  %11 = hlfir.designate %6{"x"} <%7> (%c2:%c7999:%c1, %c1:%c120:%c1, %9)  shape %10 : (!fir.ref<!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>, !fir.shape<3>, index, index, index, index, index, index, i64, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  %12 = hlfir.elemental %10 unordered : (!fir.shape<2>) -> !hlfir.expr<7998x120xf32> {
  ^bb0(%arg1: index, %arg2: index):
    %16 = hlfir.designate %11 (%arg1, %arg2)  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
    %17 = fir.load %16 : !fir.ref<f32>
    %18 = hlfir.no_reassoc %17 : f32
    hlfir.yield_element %18 : f32
  }
  %13 = fir.load %1#0 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> i64
  %15 = hlfir.designate %6{"x"} <%7> (%c2:%c7999:%c1, %c1:%c120:%c1, %14)  shape %10 : (!fir.ref<!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>, !fir.shape<3>, index, index, index, index, index, index, i64, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  hlfir.assign %12 to %15 : !hlfir.expr<7998x120xf32>, !fir.box<!fir.array<7998x120xf32>>
  hlfir.destroy %12 : !hlfir.expr<7998x120xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK:           fir.do_loop %[[VAL_24:.*]] =
// CHECK:             fir.do_loop %[[VAL_25:.*]] =
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_20:.*]] (%[[VAL_25]], %[[VAL_24]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<f32>
// CHECK:               %[[VAL_28:.*]] = hlfir.no_reassoc %[[VAL_27]] : f32
// CHECK:               %[[VAL_29:.*]] = hlfir.designate %[[VAL_23:.*]] (%[[VAL_25]], %[[VAL_24]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_28]] to %[[VAL_29]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }
