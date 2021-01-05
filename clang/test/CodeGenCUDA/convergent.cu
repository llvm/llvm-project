// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -emit-llvm \
// RUN:   -disable-llvm-passes -o - %s | FileCheck -allow-deprecated-dag-overlap -check-prefix DEVICE %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -disable-llvm-passes -o - %s | \
// RUN:  FileCheck -allow-deprecated-dag-overlap -check-prefix HOST %s

#include "Inputs/cuda.h"

// DEVICE: Function Attrs:
// DEVICE-SAME: convergent
// DEVICE-NEXT: define{{.*}} void @_Z3foov
__device__ void foo() {}

// HOST: Function Attrs:
// HOST-NOT: convergent
// HOST-NEXT: define{{.*}} void @_Z3barv
// DEVICE: Function Attrs:
// DEVICE-SAME: convergent
// DEVICE-NEXT: define{{.*}} void @_Z3barv
__host__ __device__ void baz();
__host__ __device__ void bar() {
  // DEVICE: call void @_Z3bazv() [[CALL_ATTR:#[0-9]+]]
  baz();
  // DEVICE: call i32 asm "trap;", "=l"() [[ASM_ATTR:#[0-9]+]]
  int x;
  asm ("trap;" : "=l"(x));
  // DEVICE: call void asm sideeffect "trap;", ""() [[ASM_ATTR:#[0-9]+]]
  asm volatile ("trap;");
}

// DEVICE: declare void @_Z3bazv() [[BAZ_ATTR:#[0-9]+]]
// DEVICE: attributes [[BAZ_ATTR]] = {
// DEVICE-SAME: convergent
// DEVICE-SAME: }
// DEVICE-DAG: attributes [[CALL_ATTR]] = { convergent
// DEVICE-DAG: attributes [[ASM_ATTR]] = { convergent

// HOST: declare void @_Z3bazv() [[BAZ_ATTR:#[0-9]+]]
// HOST: attributes [[BAZ_ATTR]] = {
// HOST-NOT: convergent
// NOST-SAME: }
