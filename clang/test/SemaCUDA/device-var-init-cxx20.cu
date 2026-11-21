// REQUIRES: nvptx-registered-target

// C++20 cases split out of device-var-init.cu.

// RUN: %clang_cc1 -verify %s -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++20
// RUN: %clang_cc1 -verify %s -std=c++20

#include "Inputs/cuda.h"

struct CE_NED {
  int x;
  constexpr CE_NED() { x = 43; }
  constexpr ~CE_NED() { x = 0; }
};

__device__ void df_local_static_constexpr() {
  static constexpr CE_NED ce;
  static constinit CE_NED ci;
  // expected-error@-1 {{cannot use 'static' local variable requiring runtime initialization or destruction in __device__ function}}
  static CE_NED ned;
  // expected-error@-1 {{cannot use 'static' local variable requiring runtime initialization or destruction in __device__ function}}
}

__host__ __device__ void hd_local_static_constexpr() {
  static constexpr CE_NED ce;
}
