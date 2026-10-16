// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:   -foffload-implicit-host-device-templates -std=c++14 \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device \
// RUN:   -foffload-implicit-host-device-templates -std=c++14 \
// RUN:   -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__host__ __device__ constexpr int pick_constexpr_unused(long);
__host__ __device__ constexpr int pick_constexpr_unused(unsigned long);

constexpr int constexpr_unused(int x) {
  return pick_constexpr_unused(x);
}

__host__ __device__ constexpr int pick_constexpr_used(long);
// expected-note@-1 {{candidate function}}
__host__ __device__ constexpr int pick_constexpr_used(unsigned long);
// expected-note@-1 {{candidate function}}

constexpr int constexpr_used(int x) {
  return pick_constexpr_used(x);
  // expected-error@-1 {{call to 'pick_constexpr_used' is ambiguous}}
}

__host__ __device__ int pick_template_unused(long);
__host__ __device__ int pick_template_unused(unsigned long);

template <typename T> int template_unused(T x) {
  return pick_template_unused(x);
}

void host_only() {
  (void)constexpr_unused(1);
  (void)template_unused(1);
}

__host__ __device__ int pick_template_used(long);
// expected-note@-1 {{candidate function}}
__host__ __device__ int pick_template_used(unsigned long);
// expected-note@-1 {{candidate function}}

template <typename T> int template_used(T x) {
  return pick_template_used(x);
  // expected-error@-1 {{call to 'pick_template_used' is ambiguous}}
}

__device__ int device_caller() {
  return constexpr_used(1) + template_used(1);
  // expected-note@-1 {{called by 'device_caller'}}
  // expected-note@-2 {{called by 'device_caller'}}
}

__global__ void kernel(int *out) {
  *out = device_caller();
  // expected-note@-1 {{called by 'kernel'}}
  // expected-note@-2 {{called by 'kernel'}}
}
