// Test FLOP counting with vector operations
//
// This test verifies that the FLOP counter correctly counts vector
// floating-point operations.
//
// RUN: %clangxx -O2 -g -mllvm -enable-instrumentor -mllvm -instrumentor-read-config-files=%config_dir/flop-counter/flop_counter_config.json %s -L%lib_dir -l%flop_counter_lib -o %t
// RUN: %t | FileCheck %s
//
// CHECK: Total FLOPs: 1000
// CHECK: Vector FLOPs: 1000

#include <cmath>
#include <stdio.h>

// Function using vector operations (if vectorized by the compiler)
void vector_compute(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = std::sqrt(a[i] * a[i] + b[i] * b[i]);
  }
}

int main(void) {
  const int N = 1000;
  float a[N], b[N], c[N];

  // Initialize arrays
  for (int i = 0; i < N; i++) {
    a[i] = (float)i;
    b[i] = (float)(i + 1);
  }

  // Compute
  vector_compute(a, b, c, N);

  // Prevent optimization
  float sum = 0.0f;
  for (int i = 0; i < N; i++) {
    sum += c[i];
  }

  if (sum > 0.0f) {
    printf("Vector computation complete\n");
  }

  return 0;
}
