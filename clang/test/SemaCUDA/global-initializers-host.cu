// RUN: %clang_cc1 %s --std=c++11 -triple x86_64-linux-unknown -fsyntax-only -o - -verify

#include "Inputs/cuda.h"

// Check that we get an error if we try to call a __device__ function from a
// module initializer.

struct S {
  __device__ S() {}
  // expected-note@-1 {{'S' declared here}}
};

S s;
// expected-error@-1 {{reference to __device__ function 'S' in global initializer}}

struct T {
  __host__ __device__ T() {}
};
T t;  // No error, this is OK.

struct U {
  __host__ U() {}
  __device__ U(int) {}
  // expected-note@-1 {{'U' declared here}}
};
U u(42);
// expected-error@-1 {{reference to __device__ function 'U' in global initializer}}

__device__ int device_fn() { return 42; }
// expected-note@-1 {{'device_fn' declared here}}
int n = device_fn();
// expected-error@-1 {{reference to __device__ function 'device_fn' in global initializer}}
