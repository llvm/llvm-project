// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -verify=dev,expected -fsyntax-only \
// RUN:   -verify-ignore-unexpected=warning -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only \
// RUN:   -verify-ignore-unexpected=warning -verify-ignore-unexpected=note %s

#include "Inputs/cuda.h"

__device__ void device_fn() {
  auto f1 = [&] {};
  f1(); // implicitly __device__

  auto f2 = [&] __device__ {};
  f2();

  auto f3 = [&] __host__ {};
  f3();  // expected-error {{no matching function}}

  auto f4 = [&] __host__ __device__ {};
  f4();

  // Now do it all again with '()'s in the lambda declarations: This is a
  // different parse path.
  auto g1 = [&]() {};
  g1(); // implicitly __device__

  auto g2 = [&]() __device__ {};
  g2();

  auto g3 = [&]() __host__ {};
  g3();  // expected-error {{no matching function}}

  auto g4 = [&]() __host__ __device__ {};
  g4();

  // Once more, with the '()'s in a different place.
  auto h1 = [&]() {};
  h1(); // implicitly __device__

  auto h2 = [&] __device__ () {};
  h2();

  auto h3 = [&] __host__ () {};
  h3();  // expected-error {{no matching function}}

  auto h4 = [&] __host__ __device__ () {};
  h4();
}

// Behaves identically to device_fn.
__global__ void kernel_fn() {
  auto f1 = [&] {};
  f1(); // implicitly __device__

  auto f2 = [&] __device__ {};
  f2();

  auto f3 = [&] __host__ {};
  f3();  // expected-error {{no matching function}}

  auto f4 = [&] __host__ __device__ {};
  f4();

  // No need to re-test all the parser contortions we test in the device
  // function.
}

__host__ void host_fn() {
  auto f1 = [&] {};
  f1(); // implicitly __host__ (i.e., no magic)

  auto f2 = [&] __device__ {};
  f2();  // expected-error {{no matching function}}

  auto f3 = [&] __host__ {};
  f3();

  auto f4 = [&] __host__ __device__ {};
  f4();
}

__host__ __device__ void hd_fn() {
  auto f1 = [&] {};
  f1(); // implicitly __host__ __device__

  auto f2 = [&] __device__ {};
  f2();
#ifndef __CUDA_ARCH__
  // expected-error@-2 {{reference to __device__ function}}
#endif

  auto f3 = [&] __host__ {};
  f3();
#ifdef __CUDA_ARCH__
  // expected-error@-2 {{reference to __host__ function}}
#endif

  auto f4 = [&] __host__ __device__ {};
  f4();
}

// The special treatment above only applies to lambdas.
__device__ void foo() {
  struct X {
    void foo() {}
  };
  X x;
  x.foo(); // dev-error {{reference to __host__ function 'foo' in __device__ function}}
}
