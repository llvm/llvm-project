// Test offload registration for two targets, and test offload target validation.
// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-pc-linux-gnu,powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=aarch64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

void foo() {
#pragma omp target
  {}
}

// CHECK-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i[[SZ:32|64]], i32, i32 }

// Check target registration is registered as a Ctor.
// CHECK: appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* @.omp_offloading.requires_reg, i8* null }]

// Check presence of foo() and the outlined target region
// CHECK: define{{.*}} void [[FOO:@.+]]()
// CHECK: define internal void [[OUTLINEDTARGET:@.+]]()

// Check registration and unregistration code.

// CHECK:     define internal void @.omp_offloading.requires_reg()
// CHECK:     call void @__tgt_register_requires(i64 1)
// CHECK:     ret void
