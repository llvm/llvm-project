// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note -DDEVICE %s

// Check that we can reference (get a function pointer to) a __global__
// function from the host side, but not the device side.  (We don't yet support
// device-side kernel launches.)

#include "Inputs/cuda.h"

struct Dummy {};

__global__ void kernel() {}

typedef void (*fn_ptr_t)();

__host__ __device__ fn_ptr_t get_ptr_hd() {
  return kernel;
#ifdef DEVICE
  // expected-error@-2 {{reference to __global__ function}}
#endif
}
__host__ fn_ptr_t get_ptr_h() {
  return kernel;
}
__device__ fn_ptr_t get_ptr_d() {
  return kernel;  // expected-error {{reference to __global__ function}}
}
