// RUN: %clang_cc1 -std=c++17 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -fcuda-is-device -verify %s
// RUN: %clang_cc1 -std=c++17 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -verify %s

#include "Inputs/cuda.h"

template <typename T>
struct S {
  __host__ __device__ S(T);
};

template <typename T>
S(T) -> S<T>;

// CUDA/HIP target attributes on deduction guides are rejected.
template <typename U>
__host__ S(U) -> S<U>;   // expected-error {{in CUDA/HIP, target attributes are not allowed on deduction guides; deduction guides are implicitly enabled for both host and device}}

template <typename V>
__device__ S(V) -> S<V>; // expected-error {{in CUDA/HIP, target attributes are not allowed on deduction guides; deduction guides are implicitly enabled for both host and device}}

template <typename W>
__global__ S(W) -> S<W>; // expected-error {{in CUDA/HIP, target attributes are not allowed on deduction guides; deduction guides are implicitly enabled for both host and device}}
