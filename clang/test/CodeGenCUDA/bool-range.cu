// RUN: %clang_cc1 -emit-llvm %s -O3 -o - -fcuda-is-device \
// RUN:   -triple nvptx64-unknown-unknown | FileCheck %s -check-prefixes=NV
// RUN: %clang_cc1 -emit-llvm %s -O3 -o - -fcuda-is-device \
// RUN:   -triple amdgcn-amd-amdhsa | FileCheck %s -check-prefixes=AMD

#include "Inputs/cuda.h"

// NV:  %[[LD:[0-9]+]] = load i8, ptr %x,{{.*}} !range ![[MD:[0-9]+]]
// NV:  store i8 %[[LD]], ptr %y
// NV: ![[MD]] = !{i8 0, i8 2}

// Make sure bool loaded from memory is truncated and
// range metadata is not emitted.
// TODO: Re-enable range metadata after issue
// https://github.com/llvm/llvm-project/issues/58176 is fixed.

// AMD:  %[[LD:[0-9]+]] = load i8, ptr addrspace(1) %x.global
// AMD-NOT: !range
// AMD:  %[[AND:[0-9]+]] = and i8 %[[LD]], 1
// AMD:  store i8 %[[AND]], ptr addrspace(1) %y.global
__global__ void test1(bool *x, bool *y) {
  *y = *x != false;
}
