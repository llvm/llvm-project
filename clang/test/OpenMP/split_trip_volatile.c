// Volatile trip count — IR shows `load volatile` of bound + split IVs (omp_fill segment).
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

volatile int n;

// CHECK-LABEL: define {{.*}} @f
// CHECK: load volatile i32, ptr @n
// CHECK: .split.iv
void f(void) {
#pragma omp split counts(2, omp_fill)
  for (int i = 0; i < n; ++i) {
  }
}
