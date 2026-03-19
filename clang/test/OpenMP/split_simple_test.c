/*
 * Simple test for #pragma omp split counts: one for-loop is transformed
 * into two loops (counts(5, 5) => [0..5) and [5..10)).
 */
// Verify the split directive compiles and emits IR (two sequential loops).
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int main(void) {
  const int n = 10;
  int sum = 0;

#pragma omp split counts(5, 5)
  for (int i = 0; i < n; ++i) {
    sum += i;
  }

  return (sum == n * (n - 1) / 2) ? 0 : 1;
}

// CHECK: define
// CHECK: load
// Split produces two sequential loops (counts(5, 5) => bounds 5, 10).
// CHECK: .split.iv
// CHECK: icmp slt i32 {{.*}}, 5
// CHECK: .split.iv
// CHECK: icmp slt i32 {{.*}}, 10
// CHECK: br i1
