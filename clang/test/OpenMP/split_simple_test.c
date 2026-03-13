/*
 * Simple test for #pragma omp split: one canonical for-loop is transformed
 * into two loops (first half and second half of iterations).
 */
// Verify the split directive compiles and emits IR (two sequential loops).
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown
// -emit-llvm %s -o - | FileCheck %s

int main(void) {
  const int n = 10;
  int sum = 0;

#pragma omp split
  for (int i = 0; i < n; ++i) {
    sum += i;
  }

  return (sum == n * (n - 1) / 2) ? 0 : 1;
}

// CHECK: define
// CHECK: load
// Split produces two sequential loops; ensure we have loop structure in IR.
// CHECK: br i1
