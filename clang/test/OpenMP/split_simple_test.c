/*
 * Simple test for #pragma omp split: one canonical for-loop is transformed
 * into two loops (first half and second half of iterations). This file
 * verifies compilation and correct result at runtime.
 *
 * Compile: clang -fopenmp -fopenmp-version=60 -o split_simple_test split_simple_test.c
 * Run:     ./split_simple_test
 * Expected: prints "sum 0..9 = 45 (expected 45)", exit code 0.
 */
// Verify the split directive compiles and links.
// RUN: %clang -fopenmp -fopenmp-version=60 -o %t %s

#include <stdio.h>

int main(void) {
  const int n = 10;
  int sum = 0;

#pragma omp split
  for (int i = 0; i < n; ++i) {
    sum += i;
  }

  printf("sum 0..%d = %d (expected %d)\n", n - 1, sum, n * (n - 1) / 2);
  return (sum == n * (n - 1) / 2) ? 0 : 1;
}
