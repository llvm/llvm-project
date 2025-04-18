// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -fconvergence-control -Wno-unused-value %s -o - 2>&1

#define __device__ __attribute__((device))

__device__ void foo() __attribute__((convergent));

__device__ int annotated_target(int cond) {
  int i = 0;

  if (cond > 1) {
    goto jumptarget;
  }

  while (true) {
    foo();
  jumptarget: [[clang::noconvergent]] {
      foo();
      i++;
    }
  }

  return i;
}

__device__ int jump_across(int cond) {
  int i = 0;

  if (cond > 1) {
    goto jumptarget;
  }

  while (true) [[clang::noconvergent]] {
    foo();
    i++;
  }

  jumptarget:
  return i;
}

__device__ int jump_internal(int cond) {
  int i = 0;

  while (true) [[clang::noconvergent]] {
    foo();
  jumptarget:
    i++;

    if (cond > 1) {
      goto jumptarget;
    }
  }
  return i;
}

__device__ void nest_to_nest(int cond) {
  int i = 0;

  for (;;) {
    foo();
    if (cond != 0) {
      [[clang::noconvergent]] while (true) {
        foo();
        if (cond > 1) {
          goto jumptarget;
        }
      }
    }
  }

  while (true) {
    foo();
    if (cond > 1) {
      do {
        foo();
        i++;
      jumptarget:
        i++;
      } while (true);
    }
  }
}
