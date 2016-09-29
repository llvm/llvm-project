// RUN: %clang_cc1 -fcuda-is-device -verify -S %s -o /dev/null
// RUN: %clang_cc1 -verify -DHOST %s -S -o /dev/null

#include "Inputs/cuda.h"

#ifdef HOST
// expected-no-diagnostics
#endif

__host__ __device__ void hd(int n) {
  int x[n];
#ifndef HOST
  // expected-error@-2 {{cannot use variable-length arrays in __host__ __device__ functions}}
#endif
}

// No error because never codegen'ed for device.
__host__ __device__ inline void hd_inline(int n) {
  int x[n];
}
void call_hd_inline() { hd_inline(42); }
