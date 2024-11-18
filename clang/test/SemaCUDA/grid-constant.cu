// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s
#include "Inputs/cuda.h"

struct S {};

__global__ void kernel_struct(__grid_constant__ const S arg) {}
__global__ void kernel_scalar(__grid_constant__ const int arg) {}

__global__ void gc_kernel_non_const(__grid_constant__ S arg) {} // expected-error {{__grid_constant__ is only allowed on const-qualified kernel parameters}}

void non_kernel(__grid_constant__ S arg) {} // expected-error {{__grid_constant__ is only allowed on const-qualified kernel parameters}}

// templates w/ non-dependent argument types get diagnosed right
// away, without instantiation.
template <typename T>
__global__ void tkernel_nd_const(__grid_constant__ const S arg, T dummy) {}
template <typename T>
__global__ void tkernel_nd_non_const(__grid_constant__ S arg, T dummy) {} // expected-error {{__grid_constant__ is only allowed on const-qualified kernel parameters}}

// dependent arguments get diagnosed after instantiation.
template <typename T>
__global__ void tkernel_const(__grid_constant__ const T arg) {}

template <typename T>
__global__ void tkernel(__grid_constant__ T arg) {} // expected-error {{__grid_constant__ is only allowed on const-qualified kernel parameters}}

void foo() {
  tkernel_const<const S><<<1,1>>>({});
  tkernel_const<S><<<1,1>>>({});
  tkernel<const S><<<1,1>>>({});
  tkernel<S><<<1,1>>>({}); // expected-note {{in instantiation of function template specialization 'tkernel<S>' requested here}}
}
