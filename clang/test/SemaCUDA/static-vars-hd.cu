// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -S -o /dev/null -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -S -o /dev/null -D HOST -verify %s

#include "Inputs/cuda.h"

#ifdef HOST
// expected-no-diagnostics
#endif

__host__ __device__ void f() {
  static int x = 42;
#ifndef HOST
  // expected-error@-2 {{within a __host__ __device__ function, only __shared__ variables may be marked 'static'}}
#endif
}

inline __host__ __device__ void g() {
  static int x = 42; // no error on device because this is never codegen'ed there.
}
void call_g() { g(); }
