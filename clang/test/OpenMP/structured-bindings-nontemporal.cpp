// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -std=c++20 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

struct Point { int x, y; };

// CHECK-LABEL: @_Z21test_simd_nontemporalv
void test_simd_nontemporal() {
  Point p{1, 2};
  auto [a, b] = p;
  int arr[10];

#pragma omp simd nontemporal(a, b)
  for (int i = 0; i < 10; ++i) {
    // CHECK: load i32,{{.*}}!nontemporal
    // CHECK: load i32,{{.*}}!nontemporal
    arr[i] = a + b;
  }
}

// CHECK-LABEL: @_Z30test_simd_nontemporal_capturedv
void test_simd_nontemporal_captured() {
  Point p{3, 4};
  auto [a, b] = p;
  int sum = 0;

#pragma omp parallel
  {
#pragma omp simd nontemporal(a)
    for (int i = 0; i < 10; ++i) {
      // CHECK: load i32,{{.*}}!nontemporal
      sum += a;
    }
  }
}
