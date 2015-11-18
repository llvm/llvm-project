// Test the Sema analysis of caller-callee relationships of host device
// functions when compiling CUDA code. There are 4 permutations of this test as
// host and device compilation are separate compilation passes, and clang has
// an option to allow host calls from host device functions. __CUDA_ARCH__ is
// defined when compiling for the device and TEST_WARN_HD when host calls are
// allowed from host device functions. So for example, if __CUDA_ARCH__ is
// defined and TEST_WARN_HD is not then device compilation is happening but
// host device functions are not allowed to call device functions.

// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -triple nvptx-unknown-cuda -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-allow-host-calls-from-host-device -verify %s -DTEST_WARN_HD
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -triple nvptx-unknown-cuda -fcuda-allow-host-calls-from-host-device -verify %s -DTEST_WARN_HD

#include "Inputs/cuda.h"

__host__ void hd1h(void);
#if defined(__CUDA_ARCH__) && !defined(TEST_WARN_HD)
// expected-note@-2 {{candidate function not viable: call to __host__ function from __host__ __device__ function}}
#endif
__device__ void hd1d(void);
#ifndef __CUDA_ARCH__
// expected-note@-2 {{candidate function not viable: call to __device__ function from __host__ __device__ function}}
#endif
__host__ void hd1hg(void);
__device__ void hd1dg(void);
#ifdef __CUDA_ARCH__
__host__ void hd1hig(void);
#if !defined(TEST_WARN_HD)
// expected-note@-2 {{candidate function not viable: call to __host__ function from __host__ __device__ function}}
#endif
#else
__device__ void hd1dig(void); // expected-note {{candidate function not viable: call to __device__ function from __host__ __device__ function}}
#endif
__host__ __device__ void hd1hd(void);
__global__ void hd1g(void); // expected-note {{'hd1g' declared here}}

__host__ __device__ void hd1(void) {
#if defined(TEST_WARN_HD) && defined(__CUDA_ARCH__)
// expected-warning@-2 {{calling __host__ function hd1h from __host__ __device__ function hd1}}
// expected-warning@-3 {{calling __host__ function hd1hig from __host__ __device__ function hd1}}
#endif
  hd1d();
#ifndef __CUDA_ARCH__
// expected-error@-2 {{no matching function}}
#endif
  hd1h();
#if defined(__CUDA_ARCH__) && !defined(TEST_WARN_HD)
// expected-error@-2 {{no matching function}}
#endif

  // No errors as guarded
#ifdef __CUDA_ARCH__
  hd1d();
#else
  hd1h();
#endif

  // Errors as incorrectly guarded
#ifndef __CUDA_ARCH__
  hd1dig(); // expected-error {{no matching function}}
#else
  hd1hig();
#ifndef TEST_WARN_HD
// expected-error@-2 {{no matching function}}
#endif
#endif

  hd1hd();
  hd1g<<<1, 1>>>(); // expected-error {{reference to __global__ function 'hd1g' in __host__ __device__ function}}
}
