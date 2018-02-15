// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s

// RUN: %clang_cc1 -fsyntax-only -fcuda-rdc -verify=rdc %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -fcuda-rdc -verify=rdc %s
// These declarations are fine in separate compilation mode:
// rdc-no-diagnostics

#include "Inputs/cuda.h"

__device__ void foo() {
  extern __shared__ int x; // expected-error {{__shared__ variable 'x' cannot be 'extern'}}
  extern __shared__ int arr[];  // ok
  extern __shared__ int arr0[0]; // expected-error {{__shared__ variable 'arr0' cannot be 'extern'}}
  extern __shared__ int arr1[1]; // expected-error {{__shared__ variable 'arr1' cannot be 'extern'}}
  extern __shared__ int* ptr ; // expected-error {{__shared__ variable 'ptr' cannot be 'extern'}}
}

__host__ __device__ void bar() {
  extern __shared__ int arr[];  // ok
  extern __shared__ int arr0[0]; // expected-error {{__shared__ variable 'arr0' cannot be 'extern'}}
  extern __shared__ int arr1[1]; // expected-error {{__shared__ variable 'arr1' cannot be 'extern'}}
  extern __shared__ int* ptr ; // expected-error {{__shared__ variable 'ptr' cannot be 'extern'}}
}

extern __shared__ int global; // expected-error {{__shared__ variable 'global' cannot be 'extern'}}
extern __shared__ int global_arr[]; // ok
extern __shared__ int global_arr1[1]; // expected-error {{__shared__ variable 'global_arr1' cannot be 'extern'}}
