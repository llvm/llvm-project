// Test precision analysis with detailed per-operation tracking
//
// This test demonstrates how the precision analysis tracks each operation
// separately by ID and shows detailed statistics.
//
// RUN: %clangxx -O0 -g -mllvm -enable-instrumentor -mllvm -instrumentor-read-config-files=%config_dir/fp-precision-analysis/fp_precision_analysis_config.json %s -L%lib_dir -l%fp_precision_analysis_lib -o %t
// RUN: %t | FileCheck %s
//
// CHECK: Floating-Point Precision Analysis Results
// CHECK: Op ID{{.*}}Total{{.*}}D->FP16{{.*}}D->F32{{.*}}D->D{{.*}}F->FP16{{.*}}F->F
// CHECK: TOTAL
// CHECK: Column Legend:
// CHECK: D->FP16:{{.*}}Double ops that can use FP16
// CHECK: Summary by Original Precision:

#include <stdio.h>

// Each of these operations will get a unique ID
// We can track their precision requirements separately

double operation_a(double x, double y) {
  // Simple addition - should work well with float
  return x + y;
}

double operation_b(double x, double y) {
  // Multiplication - should work well with float for normal ranges
  return x * y;
}

double operation_c(double x, double y) {
  // Division - might need more precision depending on values
  return x / y;
}

double operation_d(double x) {
  // Subtraction of close values - might need double precision
  double y = x + 1e-6;
  return y - x;
}

double complex_operation(double a, double b) {
  // Multiple operations in sequence
  double temp1 = a * b;     // Op 1
  double temp2 = temp1 + a; // Op 2
  double temp3 = temp2 / b; // Op 3
  return temp3;
}

int main(void) {
  double result = 0.0;

  // Execute operations multiple times
  // Each call site gets a unique operation ID
  for (int i = 1; i < 20; i++) {
    result += operation_a(i * 1.0, i * 2.0);
    result += operation_b(i * 1.5, i * 0.5);
    result += operation_c(i * 10.0, i * 2.0);
    result += operation_d(i * 100.0);
    result += complex_operation(i * 1.5, i * 2.5);
  }

  // Some operations with different value ranges
  for (int i = 1; i < 10; i++) {
    // Very small values - might need double precision
    result += operation_a(i * 1e-5, i * 1e-5);
    // Large values - might work with float
    result += operation_b(i * 1e5, i * 1e-5);
  }

  if (result != 0.0) {
    printf("Result: %.10f\n", result);
  }

  return 0;
}
