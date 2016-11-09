// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

#ifndef __CUDA_ARCH__
// expected-no-diagnostics
#endif

// When compiling for device, foo()'s call to host_fn() is an error, because
// foo() is known-emitted.
//
// The trickiness here comes from the fact that the FunctionDecl bar() sees
// foo() does not have the "inline" keyword, so we might incorrectly think that
// foo() is a priori known-emitted.  This would prevent us from marking foo()
// as known-emitted when we see the call from bar() to foo(), which would
// prevent us from emitting an error for foo()'s call to host_fn() when we
// eventually see it.

void host_fn() {}
#ifdef __CUDA_ARCH__
  // expected-note@-2 {{declared here}}
#endif

__host__ __device__ void foo();
__device__ void bar() {
  foo();
#ifdef __CUDA_ARCH__
  // expected-note@-2 {{called by 'bar'}}
#endif
}
inline __host__ __device__ void foo() {
  host_fn();
#ifdef __CUDA_ARCH__
  // expected-error@-2 {{reference to __host__ function}}
#endif
}

// This is similar to the above, except there's no error here.  This code used
// to trip an assertion due to us noticing, when emitting the definition of
// boom(), that T::operator S() was (incorrectly) considered a priori
// known-emitted.
struct S {};
struct T {
  __device__ operator S() const;
};
__device__ inline T::operator S() const { return S(); }

__device__ T t;
__device__ void boom() {
  S s = t;
}
