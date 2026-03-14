// Test hlfir.null operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test() {
  // CHECK: hlfir.null !fir.ref<none>
  %0 = hlfir.null !fir.ref<none>
  return
}
