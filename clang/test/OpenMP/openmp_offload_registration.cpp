// Test offload registration for two targets, and test offload target validation.
// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-pc-linux-gnu,powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=aarch64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

void foo(void) {
#pragma omp target
  {}
}

// CHECK-DAG: [[ENTTY:%.+]] = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }

// Check presence of foo() and the outlined target region
// CHECK: define{{.*}} void [[FOO:@.+]]()
// CHECK: define internal void [[OUTLINEDTARGET:@.+]]()
