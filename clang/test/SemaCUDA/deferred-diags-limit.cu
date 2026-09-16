// RUN: not %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only \
// RUN:   -ferror-limit 2 2>&1 %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK: cannot use 'throw' in __host__ __device__ function
// CHECK: cannot use 'throw' in __host__ __device__ function
// CHECK-NOT: cannot use 'throw' in __host__ __device__ function
// CHECK: too many errors emitted, stopping now

inline __host__ __device__ void hasInvalid1() {
  throw NULL;
}

inline __host__ __device__ void hasInvalid2() {
  throw NULL;
}

inline __host__ __device__ void hasInvalid3() {
  throw NULL;
}

__global__ void use0() {
  hasInvalid1();
  hasInvalid2();
  hasInvalid3();
}
