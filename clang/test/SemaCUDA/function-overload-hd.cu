// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -S -o /dev/null -verify \
// RUN:   -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -S -o /dev/null -fcuda-is-device \
// RUN:   -verify -verify-ignore-unexpected=note %s

#include "Inputs/cuda.h"

// FIXME: Merge into function-overload.cu once deferred errors can be emitted
// when non-deferred errors are present.

#if !defined(__CUDA_ARCH__)
//expected-no-diagnostics
#endif

typedef void (*GlobalFnPtr)();  // __global__ functions must return void.

__global__ void g() {}

__host__ __device__ void hd() {
  GlobalFnPtr fp_g = g;
#if defined(__CUDA_ARCH__)
  // expected-error@-2 {{reference to __global__ function 'g' in __host__ __device__ function}}
#endif
  g<<<0,0>>>();
#if defined(__CUDA_ARCH__)
  // expected-error@-2 {{reference to __global__ function 'g' in __host__ __device__ function}}
#endif  // __CUDA_ARCH__
}
