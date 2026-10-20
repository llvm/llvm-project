// Split on loop in constructor using member-related bound.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

extern "C" void body(int);

struct S {
  int n;
  S() : n(10) {
#pragma omp split counts(3, omp_fill)
    for (int i = 0; i < n; ++i)
      body(i);
  }
};

// CHECK-LABEL: define {{.*}} @_ZN1SC1Ev
// CHECK: .split.iv
void use_s() {
  S s;
}
