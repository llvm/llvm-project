// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

struct S {
  __host__ ~S() {}
  __device__ ~S() {}
};

__host__ __device__ void test() {
  S s;
  // This should not crash clang.
  s.~S();
}
