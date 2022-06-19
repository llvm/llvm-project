// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__global__ void g1(int x) {}

template <typename T> void t1(T arg) {
  g1<<<arg, arg>>>(1);
}

void h1(int x) {}
int h2(int x) { return 1; }

int main(void) {
  g1<<<1, 1>>>(42);
  g1(42); // expected-error {{call to global function 'g1' not configured}}
  g1<<<1>>>(42); // expected-error {{too few execution configuration arguments to kernel function call}}
  g1<<<1, 1, 0, 0, 0>>>(42); // expected-error {{too many execution configuration arguments to kernel function call}}

  t1(1);

  h1<<<1, 1>>>(42); // expected-error {{kernel call to non-global function 'h1'}}

  int (*fp)(int) = h2;
  fp<<<1, 1>>>(42); // expected-error {{must have void return type}}

  g1<<<undeclared, 1>>>(42); // expected-error {{use of undeclared identifier 'undeclared'}}
}

// Make sure we can call static member kernels.
template <typename > struct a0 {
  template <typename T> static __global__ void Call(T);
};
struct a1 {
  template <typename T> static __global__ void Call(T);
};
template <typename T> struct a2 {
  static __global__ void Call(T);
};
struct a3 {
  static __global__ void Call(int);
  static __global__ void Call(void*);
};

struct b {
  template <typename c> void d0(c arg) {
    a0<c>::Call<<<0, 0>>>(arg);
    a1::Call<<<0,0>>>(arg);
    a2<c>::Call<<<0,0>>>(arg);
    a3::Call<<<0, 0>>>(arg);
  }
  void d1(void* arg) {
    a0<void*>::Call<<<0, 0>>>(arg);
    a1::Call<<<0,0>>>(arg);
    a2<void*>::Call<<<0,0>>>(arg);
    a3::Call<<<0, 0>>>(arg);
  }
  void e() { d0(1); }
};
