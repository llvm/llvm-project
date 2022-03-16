// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__device__ void foo(int) {} // expected-note {{candidate function not viable: call to __device__ function from __host__ function}}
// expected-note@-1 {{'foo' declared here}}

int main() {
  #pragma omp parallel
  for (int i = 0; i < 100; i++)
    foo(1); // expected-error {{no matching function for call to 'foo'}}
  
  auto Lambda = []() {
    #pragma omp parallel
    for (int i = 0; i < 100; i++)
      foo(1); // expected-error {{reference to __device__ function 'foo' in __host__ __device__ function}}
    };
  Lambda(); // expected-note {{called by 'main'}}
}
