// Test precision analysis with fp16 overflow/underflow detection
//
// This test specifically exercises float operations that would overflow or
// underflow when converted to fp16, verifying that the runtime correctly
// distinguishes between input special values and lowering-induced special values.
//
// RUN: %clangxx -O0 -g -mllvm -enable-instrumentor -mllvm -instrumentor-read-config-files=%config_dir/fp-precision-analysis/fp_precision_analysis_config.json %s -L%lib_dir -l%fp_precision_analysis_lib -o %t
// RUN: %t | FileCheck %s
//
// CHECK: Floating-Point Precision Analysis Results
// CHECK: Double operations: Try Float, then FP16 if Float works
// CHECK: Op ID{{.*}}Total{{.*}}D->FP16{{.*}}D->F32{{.*}}D->D{{.*}}F->FP16{{.*}}F->F{{.*}}InpNaN{{.*}}D-OvFl{{.*}}F-OvFl
// CHECK: D-OvFl:{{.*}}Double ops where lowering caused overflow
// CHECK: F-OvFl:{{.*}}Float ops where lowering to FP16 caused overflow

#include <math.h>
#include <stdio.h>

// Float operations with values that work in fp16 range
// fp16 max is about 65504
float small_float_ops(float a, float b) {
  // These should be fine in fp16
  return a + b;
}

// Float operations that will overflow in fp16
float large_float_ops(float a, float b) {
  // fp16 max is ~65504, these will overflow to inf
  return a * b;
}

// Float operations that will underflow in fp16
// fp16 min normal is about 6.1e-5
float tiny_float_ops(float a, float b) {
  // These will underflow to zero in fp16
  return a * b;
}

// Operations with actual NaN/Inf inputs
float special_input_ops(float a, float b) {
  // These have special values in inputs
  return a / b;
}

// Double operations with large values
double large_double_ops(double a, double b) {
  // float max is about 3.4e38, these will overflow
  return a * b;
}

int main(void) {
  float result_f = 0.0f;
  double result_d = 0.0;

  // Small float operations (should work in fp16)
  for (int i = 1; i < 20; i++) {
    result_f += small_float_ops(i * 1.5f, i * 2.5f);
  }

  // Large float operations (will overflow to inf in fp16)
  for (int i = 1; i < 15; i++) {
    float big = 10000.0f * i;
    result_f += large_float_ops(big, big); // Result > 65504
  }

  // Tiny float operations (will underflow to 0 in fp16)
  for (int i = 1; i < 15; i++) {
    float tiny = 1e-4f / i;
    result_f += tiny_float_ops(tiny, tiny); // Result < 6e-5
  }

  // Operations with NaN/Inf inputs
  result_f += special_input_ops(1.0f, 0.0f); // Inf
  result_f += special_input_ops(0.0f, 0.0f); // NaN

  // Double operations that overflow in float
  for (int i = 1; i < 10; i++) {
    double huge = 1e38 * i;
    result_d += large_double_ops(huge, huge); // Result > float_max
  }

  // Some normal double operations
  for (int i = 1; i < 30; i++) {
    result_d += i * 1.5 + i * 2.5;
  }

  if (!isnan(result_f) && !isnan(result_d)) {
    printf("Computation complete\n");
  }

  return 0;
}
