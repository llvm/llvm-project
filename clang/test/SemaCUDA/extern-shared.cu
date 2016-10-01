// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

__device__ void foo() {
  extern __shared__ int x; // expected-error {{__shared__ variable 'x' cannot be 'extern'}}
}

__host__ __device__ void bar() {
  extern __shared__ int x; // expected-error {{__shared__ variable 'x' cannot be 'extern'}}
}

extern __shared__ int global; // expected-error {{__shared__ variable 'global' cannot be 'extern'}}
