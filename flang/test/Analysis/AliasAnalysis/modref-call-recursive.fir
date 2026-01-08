// RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// Test modref for fir.call's nested in regions.

// CHECK-LABEL: Testing : "_QPtest_if_then"
// CHECK: test_if -> test_var_if#0: NoModRef
// CHECK: test_if -> test_var_if#1: NoModRef
func.func @_QPtest_if_then(%arg0: !fir.ref<f32> {fir.bindc_name = "test_var_if"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_if_thenEcond"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %2:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_if_thenEtest_var_if", test.ptr ="test_var_if"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = fir.load %1#0 : !fir.ref<!fir.logical<4>>
  %4 = fir.convert %3 : (!fir.logical<4>) -> i1
  fir.if %4 {
    fir.call @_QPtest_effect_external() fastmath<contract> : () -> ()
  } {test.ptr ="test_if"}
  return
}

// CHECK-LABEL: Testing : "_QPtest_if_then_else"
// CHECK: test_if_then_else -> test_var_if_then_else#0: NoModRef
// CHECK: test_if_then_else -> test_var_if_then_else#1: NoModRef
func.func @_QPtest_if_then_else(%arg0: !fir.ref<f32> {fir.bindc_name = "test_var_if_then_else"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_if_then_elseEcond"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %2:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_if_then_elseEtest_var_if_then_else", test.ptr ="test_var_if_then_else"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = fir.load %1#0 : !fir.ref<!fir.logical<4>>
  %4 = fir.convert %3 : (!fir.logical<4>) -> i1
  fir.if %4 {
  } else {
    fir.call @_QPtest_effect_external() fastmath<contract> : () -> ()
  } {test.ptr ="test_if_then_else"}
  return
}

// CHECK-LABEL: Testing : "_QPtest_do_loop"
// CHECK: test_do_loop -> test_var_do_loop#0: NoModRef
// CHECK: test_do_loop -> test_var_do_loop#1: NoModRef
func.func @_QPtest_do_loop(%arg0: !fir.ref<f32> {fir.bindc_name = "test_var_do_loop"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_do_loopEi"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtest_do_loopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_do_loopEtest_var_do_loop", test.ptr ="test_var_do_loop"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %c1_i32 = arith.constant 1 : i32
  %4 = fir.convert %c1_i32 : (i32) -> index
  %c10_i32 = arith.constant 10 : i32
  %5 = fir.convert %c10_i32 : (i32) -> index
  %c1 = arith.constant 1 : index
  %6 = fir.convert %4 : (index) -> i32
  %7 = fir.do_loop %arg1 = %4 to %5 step %c1 iter_args(%arg2 = %6) -> (i32) attributes {test.ptr ="test_do_loop"} {
    fir.store %arg2 to %2#0 : !fir.ref<i32>
    fir.call @_QPtest_effect_external() fastmath<contract> : () -> ()
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.load %2#0 : !fir.ref<i32>
    %10 = arith.addi %9, %8 overflow<nsw> : i32
    fir.result %10 : i32
  }
  fir.store %7 to %2#0 : !fir.ref<i32>
  return
}
func.func private @_QPtest_effect_external()
