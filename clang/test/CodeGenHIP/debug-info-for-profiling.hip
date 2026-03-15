// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -debug-info-kind=constructor -fdebug-info-for-profiling -gheterogeneous-dwarf=diexpression -o - %s 2>&1 | FileCheck %s

// Regression test for workaround in SWDEV-469667

#define __device__ __attribute__((device))

struct S {
  int member;
};

__device__ int *sink;

__device__ void kernel1(struct S *s) {
// CHECK-NOT: MDNode incompatible with Debug Info Version
  *sink = s->member;
}

