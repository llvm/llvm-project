// RUN: fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:   --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// A null address names no object, so a call cannot read or write it.
// This holds when the null is not an argument and when it is passed.

// CHECK-LABEL: Testing : "_QPtest_null_user_var"
// CHECK-DAG: other -> x#0: NoModRef
// CHECK-DAG: takes_x -> x#0: NoModRef

func.func @_QPtest_null_user_var() {
  %abs = fir.absent !fir.ref<f32>
  %0 = fir.declare %abs uniq_name("_QFtest_null_user_varEx") {test.ptr = "x"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.call @_QPother() {test.ptr = "other"} : () -> ()
  fir.call @_QPtakes_x(%0) {test.ptr = "takes_x"} : (!fir.ref<f32>) -> ()
  return
}
func.func private @_QPother()
func.func private @_QPtakes_x(%arg0: !fir.ref<f32>)
