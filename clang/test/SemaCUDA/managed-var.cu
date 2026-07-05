// RUN: %clang_cc1 -fsyntax-only -verify -x hip %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify -x hip %s
// RUN: %clang_cc1 -fsyntax-only -fgpu-rdc -verify -x hip %s
// RUN: %clang_cc1 -fsyntax-only -fgpu-rdc -fcuda-is-device -verify -x hip %s

#include "Inputs/cuda.h"

struct A {
  int a;
  A() { a = 1; }
};

__managed__ int m1;

__managed__ __managed__ int m2;

__managed__ __device__ int m3;
__device__ __managed__ int m3a;

__managed__ __constant__ int m4;
// expected-error@-1 {{'constant' and 'managed' attributes are not compatible}}
// expected-note@-2 {{conflicting attribute is here}}

__constant__ __managed__ int m4a;
// expected-error@-1 {{'managed' and 'constant' attributes are not compatible}}
// expected-note@-2 {{conflicting attribute is here}}

__managed__ __shared__ int m5;
// expected-error@-1 {{'shared' and 'managed' attributes are not compatible}}
// expected-note@-2 {{conflicting attribute is here}}

__shared__ __managed__ int m5a;
// expected-error@-1 {{'managed' and 'shared' attributes are not compatible}}
// expected-note@-2 {{conflicting attribute is here}}

__managed__ __global__ int m6;
// expected-warning@-1 {{'global' attribute only applies to functions}}

void func() {
  __managed__ int m7;
  // expected-error@-1 {{__constant__, __device__, and __managed__ are not allowed on non-static local variables}}
}

__attribute__((managed(1))) int m8;
// expected-error@-1 {{'managed' attribute takes no arguments}}

__managed__ void func2() {}
// expected-warning@-1 {{'managed' attribute only applies to variables}}

typedef __managed__ int managed_int;
// expected-warning@-1 {{'managed' attribute only applies to variables}}

__managed__ A a;
// expected-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
