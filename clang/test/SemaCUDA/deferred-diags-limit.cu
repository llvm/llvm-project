// RUN: not %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only \
// RUN:   -ferror-limit 2 2>&1 %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK: cannot use 'throw' in __host__ __device__ function
// CHECK: cannot use 'throw' in __host__ __device__ function
// CHECK-NOT: cannot use 'throw' in __host__ __device__ function
// CHECK: too many errors emitted, stopping now

inline __host__ __device__ void hasInvalid() {
  throw NULL;
}

__global__ void use0() {
  hasInvalid();
  hasInvalid();
  hasInvalid();
  hasInvalid();
}
