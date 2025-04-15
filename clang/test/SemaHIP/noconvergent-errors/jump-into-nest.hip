// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 -fsyntax-only -triple amdgcn -fconvergence-control -Wno-unused-value %s -o - 2>&1 | FileCheck %s

#define __device__ __attribute__((device))

__device__ void foo() __attribute__((convergent));

__device__ int jump_into_nest(int cond) {
  int i = 0;

  if (cond > 1) {
    goto jumptarget;
  }

  foo();

  while (true) {
    foo();
    if (cond > 1) {
  // CHECK: :[[# @LINE+1]]:{{[0-9]+}}: error: cannot jump into a noconvergent statement from outside
      do [[clang::noconvergent]] {
        i++;
        foo();
      jumptarget:
        i++;
      } while (true);
      foo();
    }
  }

  return i;
}
