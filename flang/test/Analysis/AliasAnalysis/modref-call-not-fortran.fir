// RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// Test that fir.call modref is conservative when it cannot enusre it is
// dealing with a Fortran user variable or a Fortran user procedure.

// Function "unknown" is not known to be a Fortran procedure.
func.func @_QPtest_unknown_call(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {test.ptr = "x", uniq_name = "_QFtest_unknown_callEx"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  fir.call @unknown() {test.ptr = "unknown"} : () -> ()
  return
}
func.func private @unknown()
// CHECK-LABEL: Testing : "_QPtest_unknown_call"
// CHECK: unknown -> x#0: ModRef

// Address "unknown_var" cannot be related to a Fortran variable.
func.func @_QPtest_unknown_var(%arg0: !fir.ref<f32>) attributes {test.ptr = "unknown_var"} {
  fir.call @_QPfortran_procedure() {test.ptr = "fortran_procedure"}: () -> ()
  return
}
func.func private @_QPfortran_procedure()
// CHECK-LABEL: Testing : "_QPtest_unknown_var"
// CHECK: fortran_procedure -> unknown_var.region0#0: ModRef
