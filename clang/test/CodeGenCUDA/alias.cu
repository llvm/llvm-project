// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -emit-llvm \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -emit-llvm -target-sdk-version=10.1 \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -o - %s | FileCheck %s

#include "Inputs/cuda.h"

extern "C" {
__device__ int foo() { return 1; }
}

[[gnu::alias("foo")]] __device__ int alias();

// CHECK: @_Z5aliasv = alias i32 (), ptr @foo
//
//      CHECK: define dso_local i32 @foo() #[[ATTR0:[0-9]+]] {
// CHECK-NEXT: entry:
//      CHECK:   ret i32 1
// CHECK-NEXT: }

// RUN: not %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -emit-llvm -target-sdk-version=9.0 \
// RUN:   -o - %s 2>&1 | FileCheck %s --check-prefix=NO_SUPPORT
// NO_SUPPORT: CUDA older than 10.0 does not support .alias
