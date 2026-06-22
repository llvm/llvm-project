// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++20 \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++20 \
// RUN:   -fsyntax-only -verify %s

// Overload ambiguity inside an implicit-H+D explicit-inst member with an
// organic device caller surfaces the deferred error with call-stack notes.

#include "Inputs/cuda.h"

__host__ __device__ int pick(long);
// expected-note@-1 {{candidate function}}
__host__ __device__ int pick(unsigned long);
// expected-note@-1 {{candidate function}}

template <typename T>
struct ETI {
  constexpr int call(T x) { return pick(x); }
  // expected-error@-1 {{call to 'pick' is ambiguous}}
};
template class ETI<int>;

__device__ int caller(ETI<int> *p, int x) { return p->call(x); }
// expected-note@-1 {{called by 'caller'}}

__global__ void kernel(ETI<int> *p, int *out) { *out = caller(p, 1); }
// expected-note@-1 {{called by 'kernel'}}
