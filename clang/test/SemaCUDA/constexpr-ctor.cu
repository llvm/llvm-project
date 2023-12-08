// RUN: %clang_cc1 -std=c++11 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -fcuda-is-device -verify=dev %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:            -verify=host %s

// host-no-diagnostics

#include "Inputs/cuda.h"

struct A {
  A(); // dev-note 2{{'A' declared here}}
};

template <class T, int x> struct B {
  T a;
  constexpr B() = default; // dev-error 2{{reference to __host__ function 'A' in __host__ __device__ function}}
};

__host__ void f() { B<A, 1> x; }
__device__ void f() { B<A, 1> x; } // dev-note{{called by 'f'}}

struct foo {
  __host__ foo() { B<A, 2> x; }
  __device__ foo() { B<A, 2> x; } // dev-note{{called by 'foo'}}
};

__host__ void g() { foo x; }
__device__ void g() { foo x; } // dev-note{{called by 'g'}}

struct bar {
  __host__ bar() { B<A, 3> x; }
  __device__ bar() { B<A, 3> x; } // no error since no instantiation of bar
};
