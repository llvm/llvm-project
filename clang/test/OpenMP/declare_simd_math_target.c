// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes \
// RUN:   -fopenmp -ffast-math -emit-llvm %s -o - | FileCheck %s

#include "Inputs/declare-simd-math.h"

double base(double x) { return sin(x); }
__attribute__((target("avx2"))) double avx2(double x) { return sin(x); }

// CHECK-LABEL: define{{.*}} double @base(
// CHECK: call{{.*}} double @sin({{.*}}) #[[BASE:[0-9]+]]
// CHECK-LABEL: define{{.*}} double @avx2(
// CHECK: call{{.*}} double @sin({{.*}}) #[[AVX2:[0-9]+]]
// CHECK-DAG: attributes #[[BASE]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_sin"
// CHECK-DAG: attributes #[[AVX2]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_sin,_ZGVdN4v_sin"
