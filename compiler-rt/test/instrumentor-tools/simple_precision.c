// Test basic precision analysis functionality
//
// This test verifies that the precision analysis runtime correctly identifies
// operations that could use lower precision with acceptable accuracy.
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

// Simple operations with large enough values that float precision is sufficient
double simple_add(double a, double b) { return a + b; }

double simple_mul(double a, double b) { return a * b; }

double simple_div(double a, double b) { return a / b; }

// Function that uses values where precision matters more
double precise_computation(double x) {
  // These operations on small differences might need double precision
  double y = x + 1e-8;
  double z = y - x;
  return z * 1e8;
}

int main(void) {
  double result = 0.0;

  // Simple operations with "normal" range values
  // These should generally work fine with float precision
  for (int i = 0; i < 100; i++) {
    result += simple_add(i * 1.5, i * 2.5);
    result += simple_mul(i * 0.5, i * 0.5);
    if (i > 0) {
      result += simple_div(i * 10.0, i * 2.0);
    }
  }

  // Operations that might require more precision
  for (int i = 1; i < 50; i++) {
    result += precise_computation(i * 1.0);
  }

  // Prevent optimization from removing the computations
  if (result > 0.0) {
    printf("Computation complete: %f\n", result);
  }

  return 0;
}
