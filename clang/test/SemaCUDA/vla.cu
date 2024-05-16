// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -verify -Wno-vla %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -verify -DHOST -Wno-vla %s

#ifndef __CUDA_ARCH__
// expected-no-diagnostics
#endif

#include "Inputs/cuda.h"

void host(int n) {
  int x[n];
}

__device__ void device(int n) {
  int x[n];
#ifdef __CUDA_ARCH__
  // expected-error@-2 {{cannot use variable-length arrays in __device__ functions}}
#endif
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
