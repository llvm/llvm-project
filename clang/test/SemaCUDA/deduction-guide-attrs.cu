// RUN: %clang_cc1 -std=c++17 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -fcuda-is-device -verify %s
// RUN: %clang_cc1 -std=c++17 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -verify %s

#include "Inputs/cuda.h"

template <typename T>
struct S {
  __host__ __device__ S(T);
};

// A host+device deduction guide is allowed and participates in CTAD, but its
// explicit target attributes are deprecated and will be rejected in a future
// Clang version.
template <typename T>
__host__ __device__ S(T) -> S<T>; // expected-warning {{use of CUDA/HIP target attributes on deduction guides is deprecated; they will be rejected in a future version of Clang}}

__host__ __device__ void use_hd_guide() {
  S s(42); // uses the explicit __host__ __device__ deduction guide above
}

// CUDA/HIP target attributes on deduction guides are rejected when they make
// the guide host-only, device-only, or a kernel.
template <typename U>
__host__ S(U) -> S<U>;   // expected-error {{in CUDA/HIP, deduction guides may only be annotated with '__host__ __device__'; '__host__'-only, '__device__'-only, or '__global__' deduction guides are not allowed}}

template <typename V>
__device__ S(V) -> S<V>; // expected-error {{in CUDA/HIP, deduction guides may only be annotated with '__host__ __device__'; '__host__'-only, '__device__'-only, or '__global__' deduction guides are not allowed}}

template <typename W>
__global__ S(W) -> S<W>; // expected-error {{in CUDA/HIP, deduction guides may only be annotated with '__host__ __device__'; '__host__'-only, '__device__'-only, or '__global__' deduction guides are not allowed}}
