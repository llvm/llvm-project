// Split nested inside `omp parallel for` outer loop.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

extern "C" void body(int, int);

// CHECK: define {{.*}} @composition(
// CHECK: .split.iv
// CHECK: call void @body
extern "C" void composition(void) {
#pragma omp parallel for
  for (int i = 0; i < 4; ++i) {
#pragma omp split counts(2, omp_fill)
    for (int j = 0; j < 10; ++j)
      body(i, j);
  }
}
