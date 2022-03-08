// RUN: not fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test that `fir.boxproc_host` fails conversion to llvm.
// At the moment this test fails since `fir.boxproc` type does not have a conversion.

// CHECK: failed to legalize operation 'func.func'
func @test(%bproc: !fir.boxproc<(i32) -> ()>) {
  %tuple = fir.boxproc_host %bproc : (!fir.boxproc<(i32) -> ()>) -> (!fir.ref<tuple<i32,f64>>)
  return
}
