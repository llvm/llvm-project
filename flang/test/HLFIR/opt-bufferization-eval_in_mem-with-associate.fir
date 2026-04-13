// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Verify that hlfir.eval_in_mem uses the LHS array instead
// of allocating a temporary.
func.func @_QPtest() {
  %cst = arith.constant 1.000000e+00 : f32
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFtestEx"}
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %4:3 = hlfir.associate %cst {adapt.valuebyref} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
  %5 = hlfir.eval_in_mem shape %2 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg0: !fir.ref<!fir.array<10xf32>>):
    %6 = fir.call @_QParray_func(%4#0) fastmath<contract> : (!fir.ref<f32>) -> !fir.array<10xf32>
    fir.save_result %6 to %arg0(%2) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
  }
  hlfir.assign %5 to %3#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
  hlfir.end_associate %4#1, %4#2 : !fir.ref<f32>, i1
  hlfir.destroy %5 : !hlfir.expr<10xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFtestEx"}
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4:.*]]) {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
// CHECK:           %[[VAL_6:.*]]:3 = hlfir.associate %[[VAL_0]] {adapt.valuebyref} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
// CHECK:           %[[VAL_7:.*]] = fir.call @_QParray_func(%[[VAL_6]]#0) fastmath<contract> : (!fir.ref<f32>) -> !fir.array<10xf32>
// CHECK:           fir.save_result %[[VAL_7]] to %[[VAL_5]]#0(%[[VAL_4]]) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
// CHECK:           hlfir.end_associate %[[VAL_6]]#1, %[[VAL_6]]#2 : !fir.ref<f32>, i1
