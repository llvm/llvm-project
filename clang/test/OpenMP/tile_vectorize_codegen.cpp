// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fclang-abi-compat=latest -std=c++17 -fopenmp -O2 -vectorize-loops -vectorize-slp -emit-llvm %s -o - | FileCheck %s
// REQUIRES: x86-registered-target
// expected-no-diagnostics

// Why the intra-tile loop is stored min-bounded: that form carries its trip
// count in the loop bound and vectorizes, while the rectangular form a
// 'collapse' consumer needs adds a body guard that blocks the vectorizer. Both
// sides are checked here so the tradeoff is visible in one place.

// CHECK-LABEL: define {{.*}} @vec_tile(
// CHECK: fadd <4 x float>
extern "C" void vec_tile(float *a, float *b, int n) {
#pragma omp tile sizes(64)
  for (int i = 0; i < n; ++i)
    a[i] = a[i] + b[i];
}

// CHECK-LABEL: define internal void @vec_tile_collapsed.omp_outlined(
// CHECK-NOT: fadd <4 x float>
// CHECK: br i1 %{{.*}}, label %omp.body.next, label %omp.inner.for.inc
// CHECK-NOT: fadd <4 x float>
// CHECK: ret void
extern "C" void vec_tile_collapsed(float *a, float *b, int n) {
#pragma omp parallel for collapse(2)
#pragma omp tile sizes(64)
  for (int i = 0; i < n; ++i)
    a[i] = a[i] + b[i];
}
