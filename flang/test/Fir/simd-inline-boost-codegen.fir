// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s

module {
  func.func private @foo()

  func.func @test_merge_default_func_attrs() {
    // CHECK-LABEL: llvm.func @test_merge_default_func_attrs
    // CHECK: llvm.call @foo() {default_func_attrs = {existing = "1", "function-inline-threshold-bonus" = "2000"}} : () -> ()
    // CHECK-NOT: omp.simd_inline_boost
    fir.call @foo() {default_func_attrs = {existing = "1"}, omp.simd_inline_boost} : () -> ()
    return
  }
}
