// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fno-cuda-host-device-constexpr -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fno-cuda-host-device-constexpr -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

// Check that, with -fno-cuda-host-device-constexpr, constexpr functions are
// host-only, and __device__ constexpr functions are still device-only.

constexpr int f() { return 0; } // expected-note {{not viable}}
__device__ constexpr int g() { return 0; } // expected-note {{not viable}}

void __device__ foo() {
  f(); // expected-error {{no matching function}}
  g();
}

void __host__ foo() {
  f();
  g(); // expected-error {{no matching function}}
}
