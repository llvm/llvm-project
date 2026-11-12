// RUN: fir-opt %s | fir-opt | FileCheck %s

// CHECK-LABEL: @test_callop
func.func @test_callop(%arg0 : f32) {
  // CHECK: fir.call @callee() : () -> ()
  fir.call @callee() fastmath<none> : () -> ()
  // CHECK: fir.call @callee() : () -> ()
  fir.call @callee() {fastmath = #arith.fastmath<none>} : () -> ()
  // CHECK: fir.call @callee() fastmath<ninf,contract> : () -> ()
  fir.call @callee() fastmath<ninf,contract> : () -> ()
  // CHECK: fir.call @callee() fastmath<nnan,afn> : () -> ()
  fir.call @callee() {fastmath = #arith.fastmath<nnan,afn>} : () -> ()
  // CHECK: fir.call @callee() fastmath<fast> : () -> ()
  fir.call @callee() fastmath<fast> : () -> ()
  // CHECK: fir.call @callee() fastmath<fast> : () -> ()
  fir.call @callee() {fastmath = #arith.fastmath<fast>} : () -> ()
  return
}

func.func private @callee()
