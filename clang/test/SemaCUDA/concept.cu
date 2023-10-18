// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++20 -fsyntax-only -verify
// RUN: %clang_cc1 -triple x86_64 -x hip %s \
// RUN:   -std=c++20 -fsyntax-only -verify

// expected-no-diagnostics

#include "Inputs/cuda.h"

template <class T>
concept C = requires(T x) {
  func(x);
};

struct A {};
void func(A x) {}

template <C T> __global__ void kernel(T x) { }

int main() {
  A a;
  kernel<<<1,1>>>(a);
}
