// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check that it's OK for kernels to call HD functions that call device-only
// functions.

#include "Inputs/cuda.h"

__device__ void device_fn(int) {}
// expected-note@-1 {{declared here}}
// expected-note@-2 {{declared here}}

inline __host__ __device__ int hd1() {
  device_fn(0);  // expected-error {{reference to __device__ function}}
  return 0;
}

inline __host__ __device__ int hd2() {
  // No error here because hd2 is only referenced from a kernel.
  device_fn(0);
  return 0;
}

inline __host__ __device__ void hd3(int) {
  device_fn(0);  // expected-error {{reference to __device__ function 'device_fn'}}
}
inline __host__ __device__ void hd3(double) {}

inline __host__ __device__ void hd4(int) {}
inline __host__ __device__ void hd4(double) {
  device_fn(0);  // No error; this function is never called.
}

__global__ void kernel(int) { hd2(); }

template <typename T>
void launch_kernel() {
  kernel<<<0, 0>>>(T());
  hd1();
  hd3(T());
}

void host_fn() {
  launch_kernel<int>();
}
