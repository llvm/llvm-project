// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fsyntax-only -fcuda-is-device -verify %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fsyntax-only -fcuda-is-device -verify %s

// Tests that no ambiguities are diagnosed when resolving addresses of
// specialized template functions with the same overloads on host and device.

#include "Inputs/cuda.h"

template <typename T> __host__ void overload(T) {}
template <typename T> __device__ void overload(T) {}

__host__ __device__ void test_hd() {
  void (*x)(int) = overload<int>;
  void (*y)(float) = overload<float>;
}

__host__ void test_host() {
  void (*x)(int) = overload<int>;
  void (*y)(float) = overload<float>;
}
__device__ void test_device() {
  void (*x)(int) = overload<int>;
  void (*y)(float) = overload<float>;
}
