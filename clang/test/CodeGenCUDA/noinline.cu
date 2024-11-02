// Uses -O2 since the defalt -O0 option adds noinline to all functions.

// RUN: %clang_cc1 -triple nvptx-nvidia-cuda -fcuda-is-device \
// RUN:     -O2 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:     -O2 -disable-llvm-passes -emit-llvm -o - -x hip %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux \
// RUN:     -O2 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

__noinline__ __device__ __host__ void fun1() {}

__attribute__((noinline)) __device__ __host__ void fun2() {}

__attribute__((__noinline__)) __device__ __host__ void fun3() {}

[[gnu::__noinline__]] __device__ __host__ void fun4() {}

#define __noinline__ __attribute__((__noinline__))
__noinline__ __device__ __host__ void fun5() {}

__device__ __host__ void fun6() {}

// CHECK: define{{.*}}@_Z4fun1v{{.*}}#[[ATTR1:[0-9]*]]
// CHECK: define{{.*}}@_Z4fun2v{{.*}}#[[ATTR1:[0-9]*]]
// CHECK: define{{.*}}@_Z4fun3v{{.*}}#[[ATTR1:[0-9]*]]
// CHECK: define{{.*}}@_Z4fun4v{{.*}}#[[ATTR1:[0-9]*]]
// CHECK: define{{.*}}@_Z4fun5v{{.*}}#[[ATTR1:[0-9]*]]
// CHECK: define{{.*}}@_Z4fun6v{{.*}}#[[ATTR2:[0-9]*]]
// CHECK: attributes #[[ATTR1]] = {{.*}}noinline
// CHECK-NOT: attributes #[[ATTR2]] = {{.*}}noinline
