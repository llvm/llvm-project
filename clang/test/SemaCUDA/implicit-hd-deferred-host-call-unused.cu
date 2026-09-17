// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:   -foffload-implicit-host-device-templates -std=c++14 \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device \
// RUN:   -foffload-implicit-host-device-templates -std=c++14 \
// RUN:   -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__host__ constexpr int host_only_constexpr_unused() { return 1; }

constexpr int constexpr_unused(int x) {
  return x + host_only_constexpr_unused();
}

extern "C" int host_only_template_unused();

template <typename T> int template_unused(T x) {
  return x + host_only_template_unused();
}

extern "C" int host_only_forced_unused();

#pragma clang force_cuda_host_device begin
int forced_unused(int x) {
  return x + host_only_forced_unused();
}
#pragma clang force_cuda_host_device end

void host_context() {
  (void)constexpr_unused(1);
  (void)template_unused(1);
  (void)forced_unused(1);
}

__host__ constexpr int host_only_constexpr_used() { return 1; }
// expected-note@-1 {{'host_only_constexpr_used' declared here}}

constexpr int constexpr_used(int x) {
  return x + host_only_constexpr_used();
  // expected-error@-1 {{reference to __host__ function 'host_only_constexpr_used' in __host__ __device__ function}}
}

extern "C" int host_only_template_used();
// expected-note@-1 {{'host_only_template_used' declared here}}

template <typename T> int template_used(T x) {
  return x + host_only_template_used();
  // expected-error@-1 {{reference to __host__ function 'host_only_template_used' in __host__ __device__ function}}
}

extern "C" int host_only_forced_used();
// expected-note@-1 {{'host_only_forced_used' declared here}}

#pragma clang force_cuda_host_device begin
int forced_used(int x) {
  return x + host_only_forced_used();
  // expected-error@-1 {{reference to __host__ function 'host_only_forced_used' in __host__ __device__ function}}
}
#pragma clang force_cuda_host_device end

__device__ int device_caller() {
  return constexpr_used(1) + template_used(1) + forced_used(1);
  // expected-note@-1 {{called by 'device_caller'}}
  // expected-note@-2 {{called by 'device_caller'}}
  // expected-note@-3 {{called by 'device_caller'}}
}

__global__ void kernel(int *out) {
  *out = device_caller();
  // expected-note@-1 {{called by 'kernel'}}
  // expected-note@-2 {{called by 'kernel'}}
  // expected-note@-3 {{called by 'kernel'}}
}
