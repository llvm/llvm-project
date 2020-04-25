// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__device__ int a;

int main(void) {
  auto lambda_kernel = [&]__global__(){ a = 1;};
  // expected-error@-1 {{kernel function 'operator()' must be a free function or static member function}}
  lambda_kernel<<<1, 1>>>();
  return 0;
}
