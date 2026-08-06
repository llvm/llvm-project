// RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// Test that fir.call modref can recognize Fortran user procedures
// after ExternalNameConversion pass.

// CHECK-LABEL: Testing : "test_modref_"
// CHECK: callee_procedure -> xx#0: NoModRef
func.func @test_modref_() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca f32 {bindc_name = "xx", uniq_name = "_QFtest_modrefExx"}
  %2 = fir.declare %1 {test.ptr = "xx", uniq_name = "_QFtest_modrefExx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.call @callee_() {test.ptr = "callee_procedure"} : () -> ()
  return
}
func.func private @callee_() attributes {fir.internal_name = "_QPcallee"}
