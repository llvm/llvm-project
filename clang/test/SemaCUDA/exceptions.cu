// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify %s

#include "Inputs/cuda.h"

void host() {
  throw NULL;
  try {} catch(void*) {}
}
__device__ void device() {
  throw NULL;
  // expected-error@-1 {{cannot use 'throw' in __device__ function}}
  try {} catch(void*) {}
  // expected-error@-1 {{cannot use 'try' in __device__ function}}
}
__global__ void kernel() {
  throw NULL;
  // expected-error@-1 {{cannot use 'throw' in __global__ function}}
  try {} catch(void*) {}
  // expected-error@-1 {{cannot use 'try' in __global__ function}}
}
