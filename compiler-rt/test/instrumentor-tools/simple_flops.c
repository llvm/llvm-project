// Test basic FLOP counting functionality
//
// This test verifies that the FLOP counter correctly counts floating-point
// operations in a simple program.
//
// RUN: %clangxx -O0 -g -mllvm -enable-instrumentor -mllvm -instrumentor-read-config-files=%config_dir/flop-counter/flop_counter_config.json %s -L%lib_dir -l%flop_counter_lib -o %t
// RUN: %t | FileCheck %s
//
// TODO: For the correct values we need to track fmuladd calls too.
//
// CHECK: Total FLOPs: 400
// CHECK: Single (float):  100
// CHECK: Double (double): 300

#include <stdio.h>

// Simple function with known FLOP count
float compute_float(float a, float b, float c) {
  // 3 FLOPs: add, mul, add
  return a + b * c;
}

double compute_double(double a, double b) {
  // 4 FLOPs: mul, mul, add, div
  return (a * a + b * b) / 2.0;
}

int main(void) {
  float f1 = 1.0f, f2 = 2.0f, f3 = 3.0f;
  double d1 = 4.0, d2 = 5.0;

  // Call functions multiple times to get meaningful counts
  float result_f = 0.0f;
  for (int i = 0; i < 100; i++) {
    result_f += compute_float(f1, f2, f3);
  }

  double result_d = 0.0;
  for (int i = 0; i < 100; i++) {
    result_d += compute_double(d1, d2);
  }

  // Prevent optimization from removing the computations
  if (result_f > 0.0f && result_d > 0.0) {
    printf("Computation complete\n");
  }

  return 0;
}
