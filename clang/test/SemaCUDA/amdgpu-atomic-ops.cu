// RUN: %clang_cc1 %s -verify -fsyntax-only -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fnative-half-type \
// RUN:   -fnative-half-arguments-and-returns

// REQUIRES: amdgpu-registered-target

#include "Inputs/cuda.h"
#include <stdatomic.h>

__device__ _Float16 test_Flot16(_Float16 *p) {
  return __atomic_fetch_sub(p, 1.0f16, memory_order_relaxed);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to integer, pointer or supported floating point type ('_Float16 *' invalid)}}
}

__device__ __fp16 test_fp16(__fp16 *p) {
  return __atomic_fetch_sub(p, 1.0f16, memory_order_relaxed);
  // expected-error@-1 {{address argument to atomic operation must be a pointer to integer, pointer or supported floating point type ('__fp16 *' invalid)}}
}
