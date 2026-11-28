// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -std=c++20 -fsyntax-only -verify %s

// Unsupported types are diagnosed in runtime code, not in
// constant-evaluated code.

#include "Inputs/cuda.h"

constexpr __float128 q = 1; // expected-note {{'q' defined here}}
template <int N> struct S { static constexpr int value = N; };

__device__ int d() {
  constexpr int i = (int)q;
  int a = S<(int)q>::value;
  return i + a + (int)q; // expected-error {{'q' requires 128 bit size 'const __float128' type support, but target 'nvptx64-nvidia-cuda' does not support it}}
}
