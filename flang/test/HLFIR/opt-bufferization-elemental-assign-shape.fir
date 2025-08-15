// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Check that the elemental+assign are rewritten into a loop
// with "optimal" loop bounds, e.g. that we use constants
// when possible.

// CHECK-LABEL:   func.func @_QPtest1(
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK:           fir.do_loop %[[VAL_6:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_0]] unordered {
// CHECK-NOT: hlfir.assign{{.*}}array
func.func @_QPtest1(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<3xf32>> {fir.bindc_name = "y"}) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2 = fir.shape %c3 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 {uniq_name = "_QFtest1Ey"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
  %4:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%arg2: index):
    %7 = hlfir.designate %1#0 (%arg2)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %8 = fir.load %7 : !fir.ref<f32>
    %9 = arith.addf %8, %8 fastmath<contract> : f32
    hlfir.yield_element %9 : f32
  }
  hlfir.assign %6 to %3#0 : !hlfir.expr<?xf32>, !fir.ref<!fir.array<3xf32>>
  hlfir.destroy %6 : !hlfir.expr<?xf32>
  return
}

// CHECK-LABEL:   func.func @_QPtest2(
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
// CHECK:           fir.do_loop %[[VAL_6:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_0]] unordered {
// CHECK-NOT: hlfir.assign{{.*}}array
func.func @_QPtest2(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<3xf32>> {fir.bindc_name = "y"}) {
  %c3 = arith.constant 3 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2 = fir.shape %c3 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 {uniq_name = "_QFtest2Ey"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
  %4 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
  ^bb0(%arg2: index):
    %5 = hlfir.designate %3#0 (%arg2)  : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
    %6 = fir.load %5 : !fir.ref<f32>
    %7 = arith.addf %6, %6 fastmath<contract> : f32
    hlfir.yield_element %7 : f32
  }
  hlfir.assign %4 to %1#0 : !hlfir.expr<3xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %4 : !hlfir.expr<3xf32>
  return
}
