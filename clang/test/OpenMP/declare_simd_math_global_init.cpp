// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes \
// RUN:   -fopenmp -ffast-math -emit-llvm %s -o - | FileCheck %s

extern "C" {
#pragma omp declare simd notinbranch
double sin(double);
}

extern double d;
double g = sin(d);

// The global initializer has no function of its own, so the mapping must be
// built from the target features instead of crashing.
// CHECK-LABEL: define internal void @__cxx_global_var_init(
// CHECK: call{{.*}} double @sin({{.*}}) #[[MAPPED:[0-9]+]]
// CHECK-DAG: attributes #[[MAPPED]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_sin"
