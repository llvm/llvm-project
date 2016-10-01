// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -verify -verify-ignore-unexpected=note \
// RUN:   -S -o /dev/null %s
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only -verify-ignore-unexpected=note \
// RUN:   -DHOST -S -o /dev/null %s
#include "Inputs/cuda.h"

__host__ __device__ void hd_fn() {
  auto f1 = [&] {};
  f1(); // implicitly __host__ __device__

  auto f2 = [&] __device__ {};
  f2();
#ifdef HOST
  // expected-error@-2 {{reference to __device__ function}}
#endif

  auto f3 = [&] __host__ {};
  f3();
#ifndef HOST
  // expected-error@-2 {{reference to __host__ function}}
#endif

  auto f4 = [&] __host__ __device__ {};
  f4();
}


