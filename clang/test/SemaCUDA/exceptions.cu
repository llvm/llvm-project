// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify %s

#include "Inputs/cuda.h"

void host() {
  throw NULL;
  try {} catch(void*) {}
}
__device__ void device() {
  throw NULL;
  // expected-error@-1 {{cannot use 'throw' in __device__ function}}
  try {} catch(void*) {}
  // expected-error@-1 {{cannot use 'try' in __device__ function}}
}
__global__ void kernel() {
  throw NULL;
  // expected-error@-1 {{cannot use 'throw' in __global__ function}}
  try {} catch(void*) {}
  // expected-error@-1 {{cannot use 'try' in __global__ function}}
}

// Check that it's an error to use 'try' and 'throw' from a __host__ __device__
// function if and only if it's codegen'ed for device.

__host__ __device__ void hd1() {
  throw NULL;
  try {} catch(void*) {}
#ifdef __CUDA_ARCH__
  // expected-error@-3 {{cannot use 'throw' in __host__ __device__ function}}
  // expected-error@-3 {{cannot use 'try' in __host__ __device__ function}}
#endif
}

// No error, never instantiated on device.
inline __host__ __device__ void hd2() {
  throw NULL;
  try {} catch(void*) {}
}
void call_hd2() { hd2(); }

// Error, instantiated on device.
inline __host__ __device__ void hd3() {
  throw NULL;
  try {} catch(void*) {}
#ifdef __CUDA_ARCH__
  // expected-error@-3 {{cannot use 'throw' in __host__ __device__ function}}
  // expected-error@-3 {{cannot use 'try' in __host__ __device__ function}}
#endif
}

__device__ void call_hd3() { hd3(); }
#ifdef __CUDA_ARCH__
// expected-note@-2 {{called by 'call_hd3'}}
#endif
