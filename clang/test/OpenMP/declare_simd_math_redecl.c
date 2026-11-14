// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fopenmp -ffast-math \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// The declaration carrying the simd clauses can be followed by another
// declaration without them. The mapping is built from all redeclarations.
#pragma omp declare simd notinbranch
double sin(double);
double sin(double);

void loop_sin(double *restrict d, const double *restrict x, int n) {
#pragma omp simd
  for (int i = 0; i < n; i++)
    d[i] = sin(x[i]);
}

// CHECK: @llvm.compiler.used = appending global [1 x ptr] {{.*}}_ZGVbN2v_sin
// CHECK: call{{.*}}<2 x double> @_ZGVbN2v_sin(
// CHECK-DAG: attributes #[[MAPPED:[0-9]+]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_sin"
