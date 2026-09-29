// RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref-call-effects))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// Test that when MemoryEffectOpInterface is externally attached to fir.call,
// fir::AliasAnalysis::getModRef uses it to refine the result after
// getCallModRef returns ModAndRef.

// The test infrastructure attaches MemoryEffectOpInterface to fir::CallOp
// such that calls to functions starting with "test_pure" have no memory
// effects, while all other calls read and write DefaultResource.

// "test_pure_call" has no memory effects via the attached interface,
// so it cannot modify or reference the variable.
// "test_impure_call" reads and writes DefaultResource without specifying
// a value, so alias analysis conservatively reports ModRef.

func.func @test_call_effects(%arg0: !fir.ref<f32>) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 {test.ptr = "x", uniq_name = "_QFtestEx"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  fir.call @test_pure_call() {test.ptr = "pure_call"} : () -> ()
  fir.call @test_impure_call() {test.ptr = "impure_call"} : () -> ()
  return
}
func.func private @test_pure_call()
func.func private @test_impure_call()
// CHECK-LABEL: Testing : "test_call_effects"
// CHECK-DAG: pure_call -> x#0: NoModRef
// CHECK-DAG: impure_call -> x#0: ModRef
