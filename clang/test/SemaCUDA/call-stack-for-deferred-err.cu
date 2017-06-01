// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify %s

#include "Inputs/cuda.h"

// We should emit an error for hd_fn's use of a VLA.  This would have been
// legal if hd_fn were never codegen'ed on the device, so we should also print
// out a callstack showing how we determine that hd_fn is known-emitted.
//
// Compare to no-call-stack-for-deferred-err.cu.

inline __host__ __device__ void hd_fn(int n);
inline __device__ void device_fn2() { hd_fn(42); } // expected-note {{called by 'device_fn2'}}

__global__ void kernel() { device_fn2(); } // expected-note {{called by 'kernel'}}

inline __host__ __device__ void hd_fn(int n) {
  int vla[n]; // expected-error {{variable-length array}}
}
