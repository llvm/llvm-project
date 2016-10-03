// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify %s

#include "Inputs/cuda.h"

// We don't allow destructors to be overloaded.  Making this work would be a
// giant change to clang, and the use cases seem quite limited.

struct A {
  ~A() {} // expected-note {{previous declaration is here}}
  __device__ ~A() {} // expected-error {{destructor cannot be redeclared}}
};

struct B {
  __host__ ~B() {} // expected-note {{previous declaration is here}}
  __host__ __device__ ~B() {} // expected-error {{destructor cannot be redeclared}}
};

struct C {
  __host__ __device__ ~C() {} // expected-note {{previous declaration is here}}
  __host__ ~C() {} // expected-error {{destructor cannot be redeclared}}
};

struct D {
  __device__ ~D() {} // expected-note {{previous declaration is here}}
  __host__ __device__ ~D() {} // expected-error {{destructor cannot be redeclared}}
};

struct E {
  __host__ __device__ ~E() {} // expected-note {{previous declaration is here}}
  __device__ ~E() {} // expected-error {{destructor cannot be redeclared}}
};

