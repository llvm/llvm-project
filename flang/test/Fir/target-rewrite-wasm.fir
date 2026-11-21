// RUN: fir-opt --target-rewrite="target=wasm32-unknown-emscripten" %s | FileCheck %s
// RUN: fir-opt --target-rewrite="target=wasm64-unknown-emscripten" %s | FileCheck %s

// Verify the WebAssembly ABI lowering for complex arguments and results.

// CHECK-LABEL: func.func @complex_result_f64(
// CHECK-SAME: %[[RESULT:.*]]: !fir.ref<tuple<f64, f64>> {llvm.align = 8 : i32, llvm.sret = tuple<f64, f64>})
func.func @complex_result_f64() -> complex<f64> {
  %0 = fir.undefined complex<f64>
  return %0 : complex<f64>
}

// CHECK-LABEL: func.func @complex_argument_f64(
// CHECK-SAME: %[[ARG:.*]]: !fir.ref<tuple<f64, f64>> {llvm.align = 8 : i32, llvm.byval = tuple<f64, f64>})
func.func @complex_argument_f64(%arg0: complex<f64>) {
  return
}

// CHECK-LABEL: func.func @complex_result_f32(
// CHECK-SAME: %[[RESULT:.*]]: !fir.ref<tuple<f32, f32>> {llvm.align = 4 : i32, llvm.sret = tuple<f32, f32>})
func.func @complex_result_f32() -> complex<f32> {
  %0 = fir.undefined complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: func.func @complex_argument_f32(
// CHECK-SAME: %[[ARG:.*]]: !fir.ref<tuple<f32, f32>> {llvm.align = 4 : i32, llvm.byval = tuple<f32, f32>})
func.func @complex_argument_f32(%arg0: complex<f32>) {
  return
}
