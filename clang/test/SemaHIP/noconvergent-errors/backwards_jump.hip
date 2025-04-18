// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 -fsyntax-only -triple amdgcn -fconvergence-control -Wno-unused-value %s -o - 2>&1 | FileCheck %s

#define __device__ __attribute__((device))

__device__ void foo() __attribute__((convergent));

__device__ int simple_jump_backwards(int cond) {
  int i = 0;

  // CHECK: :[[# @LINE+1]]:{{[0-9]+}}: error: cannot jump into a noconvergent statement from outside
  while (true) [[clang::noconvergent]] {
    foo();
  jumptarget:
    i++;
  }

  if (cond > 1) {
    goto jumptarget;
  }

  return i;
}
