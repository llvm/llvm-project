// RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// If LocalAliasAnalysis cannot deduce NoModRef (e.g. due to
// a store unknown to it), then MemAlloc and and MemFree
// effects should be ignored in FIR AliasAnalysis.
// They used to be treated as Mod effect.

// CHECK-LABEL: Testing : "test_alloc_effect"
// CHECK: test_effect -> test_decl#0: Ref
func.func @test_alloc_effect(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>, %arg2: i1) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "arg0", test.ptr = "test_decl"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  fir.if %arg2 {
    %2 = fir.load %1 : !fir.ref<i32>
    // fir.declare has MemAlloc effect on the DebuggingResource:
    %3 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "arg1"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    fir.store %2 to %3 : !fir.ref<i32>
  } {test.ptr = "test_effect"}
  return
}

// CHECK-LABEL: Testing : "test_free_effect"
// CHECK: test_effect -> test_decl#0: Ref
func.func @test_free_effect(%arg0: !fir.ref<i32>, %arg1: !fir.heap<i32>, %arg2: i1) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "arg0", test.ptr = "test_decl"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "arg1"} : (!fir.heap<i32>, !fir.dscope) -> !fir.heap<i32>
  fir.if %arg2 {
    %3 = fir.load %1 : !fir.ref<i32>
    fir.store %3 to %2 : !fir.heap<i32>
    // fir.freemem used to report operation-wide MemFree effect,
    // i.e. without specifying the freed value.
    fir.freemem %2 : !fir.heap<i32>
  } {test.ptr = "test_effect"}
  return
}
