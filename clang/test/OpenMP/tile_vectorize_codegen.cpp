// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fclang-abi-compat=latest -std=c++17 -fopenmp -O2 -vectorize-loops -vectorize-slp -emit-llvm %s -o - | FileCheck %s
// REQUIRES: x86-registered-target
// expected-no-diagnostics

// Standalone tile stays min-bounded (no body predicate) and vectorizes.

// CHECK-LABEL: define {{.*}} @vec_tile(
// CHECK: fadd <4 x float>
// CHECK-NOT: omp.body.next
extern "C" void vec_tile(float *a, float *b, int n) {
#pragma omp tile sizes(64)
  for (int i = 0; i < n; ++i)
    a[i] = a[i] + b[i];
}
