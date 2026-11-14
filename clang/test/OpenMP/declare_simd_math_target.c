// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes \
// RUN:   -fopenmp -ffast-math -fblocks -emit-llvm %s -o - | FileCheck %s

#include "Inputs/declare-simd-math.h"

double base(double x) { return sin(x); }
__attribute__((target("avx2"))) double avx2(double x) { return sin(x); }
__attribute__((target_clones("avx512f", "default"))) double clones(double x) {
  return acosh(x);
}
__attribute__((cpu_specific(ivybridge))) double specific(double x) {
  return sin(x);
}
__attribute__((target("avx2"))) double block_call(double x) {
  double (^blk)(double) = ^(double y) { return sin(y); };
  return blk(x);
}

// CHECK-LABEL: define{{.*}} double @base(
// CHECK: call{{.*}} double @sin({{.*}}) #[[BASE:[0-9]+]]
// CHECK: declare <2 x double> @_ZGVbN2v_sin(<2 x double>) #[[VEC_ATTR:[0-9]+]]
// CHECK-LABEL: define{{.*}} double @avx2(
// CHECK: call{{.*}} double @sin({{.*}}) #[[AVX2:[0-9]+]]
// CHECK-LABEL: define{{.*}} double @clones.avx512f
// CHECK: call{{.*}} double @acosh({{.*}}) #[[CLONES_AVX512:[0-9]+]]
// CHECK-LABEL: define{{.*}} double @clones.default
// CHECK: call{{.*}} double @acosh({{.*}}) #[[CLONES_DEFAULT:[0-9]+]]
// CHECK-LABEL: define{{.*}} double @specific
// CHECK: call{{.*}} double @sin({{.*}}) #[[SPECIFIC:[0-9]+]]
// Blocks are compiled with the TU features, not the features of the function
// they are defined in, so they get the same mapping as the base target.
// CHECK-LABEL: define internal{{.*}} double @__block_call_block_invoke(
// CHECK: call{{.*}} double @sin({{.*}}) #[[BASE]]
// CHECK-DAG: attributes #[[BASE]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_sin"
// CHECK-DAG: attributes #[[AVX2]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_sin,_ZGVdN4v_sin"
// CHECK-DAG: attributes #[[CLONES_AVX512]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_acosh,_ZGVdN4v_acosh,_ZGVeN8v_acosh"
// CHECK-DAG: attributes #[[CLONES_DEFAULT]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_acosh"
// CHECK-DAG: attributes #[[SPECIFIC]] = {{.*}}"vector-function-abi-variant"="_ZGVbN2v_sin,_ZGVcN4v_sin"
// The variant declarations carry the attributes of the scalar function.
// CHECK-DAG: attributes #[[VEC_ATTR:[0-9]+]] = { nounwind willreturn memory(none) }
