// Tests that `fir.local` ops are able to provide an alloca block when required.

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

fir.declare_reduction @add_reduction_byref_box_heap_UxUxf32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> alloc {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  fir.yield(%0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
} init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>):
  %cst = arith.constant 0.000000e+00 : f32
  %0 = fir.load %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  hlfir.assign %cst to %0 : f32, !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  fir.yield(%arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
} combiner {
^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %arg1: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>):
  fir.yield(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
}

// CHECK-LABEL:   fir.declare_reduction @add_reduction_byref_box_heap_UxUxf32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> alloc {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:           fir.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)

// CHECK-LABEL:   } init {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>):
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:           fir.yield(%[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)

// CHECK-LABEL:   } combiner {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>):
// CHECK:           fir.yield(%[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
// CHECK:         }
