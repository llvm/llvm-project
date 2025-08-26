// Tests that for `boxchar` local values, we use the value yielded by the `init`
// region rather than the local allocated storage.

// RUN: fir-opt --split-input-file --simplify-fir-operations %s | FileCheck %s

fir.local {type = local} @_QFtestEx_private_boxchar_c8xU : !fir.boxchar<1> init {
^bb0(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>):
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1 = fir.allocmem !fir.char<1,?>(%0#1 : index) {bindc_name = "", uniq_name = ""}
  %2 = fir.emboxchar %1, %0#1 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  fir.yield(%2 : !fir.boxchar<1>)
} dealloc {
^bb0(%arg0: !fir.boxchar<1>):
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1 = fir.convert %0#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.heap<!fir.char<1,?>>
  fir.freemem %1 : !fir.heap<!fir.char<1,?>>
  fir.yield
}
func.func @_QPtest(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2:2 = hlfir.declare %1#0 typeparams %1#1 dummy_scope %0 {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_concurrent {
    %5 = fir.alloca i32 {bindc_name = "i"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFtestEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg1) = (%c1) to (%c10) step (%c1) local(@_QFtestEx_private_boxchar_c8xU %2#0 -> %arg2 : !fir.boxchar<1>) {
      %7 = fir.convert %arg1 : (index) -> i32
      fir.store %7 to %6#0 : !fir.ref<i32>
      %8:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    }
  }
  return
}

// CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtestEx"}
// CHECK: fir.do_loop %{{.*}} = %{{.*}} unordered {
// CHECK:   %[[X_UNBOX:.*]]:2 = fir.unboxchar %[[X_DECL:.*]]#0

// Verify that the value yielded by the `init` region is the one used through
// out the loop region rather than the local allocation.
// CHECK:   %[[LOCAL_ALLOC:.*]] = fir.allocmem !fir.char<1,?>(%[[X_UNBOX]]#1 : index)
// CHECK:   %[[LOCAL_BOX:.*]] = fir.emboxchar %[[LOCAL_ALLOC]], %[[X_UNBOX]]#1
// CHECK:   %[[LOCAL_UNBOX:.*]]:2 = fir.unboxchar %[[LOCAL_BOX]]
// CHECK:   %[[LOCAL_CVT:.*]] = fir.convert %[[LOCAL_UNBOX]]#0
// CHECK:   fir.freemem %[[LOCAL_CVT]]
// CHECK: }
