// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -x cuda -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -x hip -std=c++20 -fsyntax-only -verify %s

// Functions used only in constant evaluation are not diagnosed.

#include "Inputs/cuda.h"

void host_fn(); // expected-note {{'host_fn' declared here}}

constexpr int cf(bool b) {
  if (b)
    host_fn();
  return 1;
}

__device__ int constexpr_var_init() {
  constexpr int v = cf(false);
  return v;
}
__device__ int case_label(int x) {
  switch (x) {
  case cf(false):
    return 1;
  }
  return 0;
}

constexpr int cr(bool b) {
  if (b)
    host_fn(); // expected-error {{reference to __host__ function 'host_fn' in __host__ __device__ function}}
  return 2;
}
__device__ int runtime_call(bool b) {
  return cr(b); // expected-note {{called by 'runtime_call'}}
}
