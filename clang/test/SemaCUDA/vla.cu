// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -DHOST %s

#include "Inputs/cuda.h"

void host(int n) {
  int x[n];
}

__device__ void device(int n) {
  int x[n];  // expected-error {{cannot use variable-length arrays in __device__ functions}}
}
