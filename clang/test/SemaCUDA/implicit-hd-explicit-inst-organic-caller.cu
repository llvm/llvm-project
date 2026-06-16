// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++20 \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++20 \
// RUN:   -fsyntax-only -verify %s

// When an implicit-H+D explicit-inst member with a host-only call has an
// organic device caller, the deferred diagnostic must surface with the
// usual call-stack notes.

#include "Inputs/cuda.h"

extern "C" int host_only();
// expected-note@-1 {{'host_only' declared here}}

template <typename T>
struct ETI {
  constexpr T bad(T x) {
    return x + (T)host_only();
    // expected-error@-1 {{reference to __host__ function 'host_only' in __host__ __device__ function}}
  }
};
template class ETI<float>;

__device__ float caller(ETI<float> *p) { return p->bad(1.0f); }
// expected-note@-1 {{called by 'caller'}}

__global__ void kernel(ETI<float> *p, float *out) { *out = caller(p); }
// expected-note@-1 {{called by 'kernel'}}
