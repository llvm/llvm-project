// Test precision analysis with mixed float and double operations
//
// This test uses both float and double operations to verify that the
// precision analysis handles both types correctly.
//
// RUN: %clangxx -O0 -g -mllvm -enable-instrumentor -mllvm -instrumentor-read-config-files=%config_dir/fp-precision-analysis/fp_precision_analysis_config.json %s -L%lib_dir -l%fp_precision_analysis_lib -o %t
// RUN: %t | FileCheck %s
//
// CHECK: Floating-Point Precision Analysis Results
// CHECK: Double operations: Try Float, then FP16 if Float works
// CHECK: D->FP16{{.*}}D->F32{{.*}}D->D{{.*}}F->FP16{{.*}}F->F
// CHECK: Summary by Original Precision:

#include <math.h>
#include <stdio.h>

// Float operations (already using lower precision)
float compute_float_distance(float x1, float y1, float x2, float y2) {
  float dx = x2 - x1;
  float dy = y2 - y1;
  return sqrtf(dx * dx + dy * dy);
}

// Double operations (analyze if float would suffice)
double compute_double_distance(double x1, double y1, double x2, double y2) {
  double dx = x2 - x1;
  double dy = y2 - y1;
  return sqrt(dx * dx + dy * dy);
}

// Mixed precision computation
double mixed_computation(float a, double b) {
  // Implicit conversion from float to double
  double a_double = a;
  return a_double * b + a_double / b;
}

int main(void) {
  float float_result = 0.0f;
  double double_result = 0.0;

  // Float operations
  for (int i = 0; i < 50; i++) {
    float_result += compute_float_distance(i * 0.1f, i * 0.2f, (i + 1) * 0.1f,
                                           (i + 1) * 0.2f);
  }

  // Double operations with values that should work well in float
  for (int i = 0; i < 50; i++) {
    double_result +=
        compute_double_distance(i * 0.1, i * 0.2, (i + 1) * 0.1, (i + 1) * 0.2);
  }

  // Mixed precision
  for (int i = 1; i < 30; i++) {
    double_result += mixed_computation(i * 1.5f, i * 2.5);
  }

  // Prevent optimization
  if (float_result > 0.0f && double_result > 0.0) {
    printf("Float result: %f, Double result: %f\n", float_result,
           double_result);
  }

  return 0;
}
