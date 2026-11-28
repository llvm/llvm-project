// Verify that loop transformation directives do not crash under -fopenmp-simd
// and that the original associated loops are emitted without OpenMP runtime
// calls. Transforms are ignored in simd-only mode.
//
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp-simd -fopenmp-version=61 -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// expected-no-diagnostics

extern "C" void body(...);

// CHECK-LABEL: define {{.*}}void @flatten_simd(
// CHECK-NOT: .flatten.iv
// CHECK: call void ({{.*}}) @body(
extern "C" void flatten_simd(int n, int m) {
#pragma omp flatten
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      body(i, j);
}

// CHECK-LABEL: define {{.*}}void @tile_simd(
// CHECK: call void ({{.*}}) @body(
extern "C" void tile_simd(int n) {
#pragma omp tile sizes(2)
  for (int i = 0; i < n; ++i)
    body(i);
}

// CHECK-LABEL: define {{.*}}void @stripe_simd(
// CHECK: call void ({{.*}}) @body(
extern "C" void stripe_simd(int n) {
#pragma omp stripe sizes(2)
  for (int i = 0; i < n; ++i)
    body(i);
}

// CHECK-LABEL: define {{.*}}void @reverse_simd(
// CHECK: call void ({{.*}}) @body(
extern "C" void reverse_simd(int n) {
#pragma omp reverse
  for (int i = 0; i < n; ++i)
    body(i);
}

// CHECK-LABEL: define {{.*}}void @split_simd(
// CHECK-NOT: .split.iv
// CHECK: call void ({{.*}}) @body(
extern "C" void split_simd(int n) {
#pragma omp split counts(2, omp_fill)
  for (int i = 0; i < n; ++i)
    body(i);
}

// CHECK-LABEL: define {{.*}}void @interchange_simd(
// CHECK: call void ({{.*}}) @body(
extern "C" void interchange_simd(int n, int m) {
#pragma omp interchange
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      body(i, j);
}

// CHECK-LABEL: define {{.*}}void @fuse_simd(
// CHECK: call void ({{.*}}) @body(
// CHECK: call void ({{.*}}) @body(
extern "C" void fuse_simd(int n, int m) {
#pragma omp fuse
  {
    for (int i = 0; i < n; ++i)
      body(i);
    for (int j = 0; j < m; ++j)
      body(j);
  }
}

// CHECK-LABEL: define {{.*}}void @unroll_simd(
// CHECK: call void ({{.*}}) @body(
extern "C" void unroll_simd(int n) {
#pragma omp unroll
  for (int i = 0; i < n; ++i)
    body(i);
}
