// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

__device__ float f0(float) __attribute__((noconvergent));
__device__ __attribute__((noconvergent)) float f1(float);
[[clang::noconvergent]] __device__ float f2(float);

__device__ [[clang::noconvergent(1)]] float f3(float);
// expected-error@-1 {{'noconvergent' attribute takes no arguments}}

__device__ [[clang::noconvergent]] float g0;
// expected-warning@-1 {{'noconvergent' attribute only applies to functions and statements}}

__device__ __attribute__((convergent)) __attribute__((noconvergent)) float f4(float);
// expected-error@-1 {{'noconvergent' and 'convergent' attributes are not compatible}}
// expected-note@-2 {{conflicting attribute is here}}

__device__ [[clang::noconvergent]] float f5(float);
__device__ [[clang::convergent]] float f5(float);
// expected-error@-1 {{'convergent' and 'noconvergent' attributes are not compatible}}
// expected-note@-3 {{conflicting attribute is here}}

__device__ float f5(float x) {
  [[clang::noconvergent]] float y;
// expected-warning@-1 {{'noconvergent' attribute only applies to functions and statements}}

  float z;

  [[clang::noconvergent]] z = 1;
// expected-warning@-1 {{'noconvergent' attribute is ignored because there exists no call expression inside the statement}}

  [[clang::noconvergent]] z = f0(x);
}
