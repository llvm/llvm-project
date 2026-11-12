// Test hlfir.null code generation to FIR

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @test() {
  // CHECK: fir.zero_bits !fir.ref<none>
  %0 = hlfir.null !fir.ref<none>
  return
}
