// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -verify %s -S -o /dev/null
// RUN: %clang_cc1 -fcxx-exceptions -verify -DHOST %s -S -o /dev/null

#include "Inputs/cuda.h"

// Check that it's an error to use 'try' and 'throw' from a __host__ __device__
// function if and only if it's codegen'ed for device.

#ifdef HOST
// expected-no-diagnostics
#endif

__host__ __device__ void hd1() {
  throw NULL;
  try {} catch(void*) {}
#ifndef HOST
  // expected-error@-3 {{cannot use 'throw' in __host__ __device__ function 'hd1'}}
  // expected-error@-3 {{cannot use 'try' in __host__ __device__ function 'hd1'}}
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
#ifndef HOST
  // expected-error@-3 {{cannot use 'throw' in __host__ __device__ function 'hd3'}}
  // expected-error@-3 {{cannot use 'try' in __host__ __device__ function 'hd3'}}
#endif
}
__device__ void call_hd3() { hd3(); }
