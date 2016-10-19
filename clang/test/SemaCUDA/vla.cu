// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -DHOST %s

#include "Inputs/cuda.h"

void host(int n) {
  int x[n];
}

__device__ void device(int n) {
  int x[n];  // expected-error {{cannot use variable-length arrays in __device__ functions}}
}

__host__ __device__ void hd(int n) {
  int x[n];
#ifdef __CUDA_ARCH__
  // expected-error@-2 {{cannot use variable-length arrays in __host__ __device__ functions}}
#endif
}

// No error because never codegen'ed for device.
__host__ __device__ inline void hd_inline(int n) {
  int x[n];
}
void call_hd_inline() { hd_inline(42); }
