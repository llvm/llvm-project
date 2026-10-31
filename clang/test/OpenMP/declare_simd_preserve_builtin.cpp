// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -ffast-math -fopenmp -emit-llvm %s -o - | FileCheck %s

double sqrt(double);

#pragma omp declare simd notinbranch
double sqrt(double);

double test(double x) {
  return sqrt(x);
}

// CHECK-LABEL: define{{.*}}@_Z4testd(
// CHECK: call{{.*}}@_Z4sqrtd(
// CHECK-NOT: llvm.sqrt
