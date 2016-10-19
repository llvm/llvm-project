// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify %s

#include "Inputs/cuda.h"

// Here we should dump an error about the VLA in device_fn, but we should not
// print a callstack indicating how device_fn becomes known-emitted, because
// it's an error to use a VLA in any __device__ function, even one that doesn't
// get emitted.

inline __device__ void device_fn(int n);
inline __device__ void device_fn2() { device_fn(42); }

__global__ void kernel() { device_fn2(); }

inline __device__ void device_fn(int n) {
  int vla[n]; // expected-error {{variable-length array}}
}
