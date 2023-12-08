// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify=dev,expected %s

#include "Inputs/cuda.h"

__host__ void h1h(void);
__device__ void h1d(void); // expected-note {{candidate function not viable: call to __device__ function from __host__ function}}
__host__ __device__ void h1hd(void);
__global__ void h1g(void);

struct h1ds { // expected-note {{requires 1 argument}}
	      // expected-note@-1 {{candidate constructor (the implicit move constructor) not viable}}
  __device__ h1ds(); // expected-note {{candidate constructor not viable: call to __device__ function from __host__ function}}
};

__host__ void h1(void) {
  h1h();
  h1d(); // expected-error {{no matching function}}
  h1hd();
  h1g<<<1, 1>>>();
  h1ds x; // expected-error {{no matching constructor}}
}

__host__ void d1h(void); // expected-note {{candidate function not viable: call to __host__ function from __device__ function}}
__device__ void d1d(void);
__host__ __device__ void d1hd(void);
__global__ void d1g(void); // dev-note {{'d1g' declared here}}

__device__ void d1(void) {
  d1h(); // expected-error {{no matching function}}
  d1d();
  d1hd();
  d1g<<<1, 1>>>(); // dev-error {{reference to __global__ function 'd1g' in __device__ function}}
}
