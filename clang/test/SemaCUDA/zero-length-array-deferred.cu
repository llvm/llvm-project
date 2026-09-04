// RUN: %clang_cc1 -fsyntax-only -x hip -fcuda-is-device -triple amdgcn \
// RUN:   -verify %s
// RUN: %clang_cc1 -fsyntax-only -x hip -fcuda-is-device -triple amdgcn \
// RUN:   -verify=device-use -DDEVICE_USE %s

#include "Inputs/cuda.h"

static __host__ __device__ int host_only_hd_array() {
  int arr[0] = {};
  return sizeof(arr);
}

static constexpr __host__ __device__ int host_only_hd_constexpr_array() {
  int arr[0] = {};
  return sizeof(arr);
}

static __host__ __device__ int host_only_hd_lambda_array() {
  auto lambda = [] {
    int arr[0] = {};
    return sizeof(arr);
  };
  return lambda();
}

static __host__ int host_caller() {
  return host_only_hd_array() + host_only_hd_constexpr_array() +
         host_only_hd_lambda_array();
}

void host_use() {
  (void)host_caller();
}

// expected-no-diagnostics

#ifdef DEVICE_USE
static __host__ __device__ int device_used_hd_array() {
  int arr[0] = {};
  // device-use-error@-1 {{zero-length arrays}}
  return sizeof(arr);
}

static constexpr __host__ __device__ int device_used_hd_constexpr_array() {
  int arr[0] = {};
  // device-use-error@-1 {{zero-length arrays}}
  return sizeof(arr);
}

static __host__ __device__ int device_used_hd_lambda_array() {
  auto lambda = [] {
    int arr[0] = {};
    // device-use-error@-1 {{zero-length arrays}}
    return sizeof(arr);
  };
  return lambda();
  // device-use-note@-1 {{called by 'device_used_hd_lambda_array'}}
}

__global__ void kernel(int *out) {
  *out = device_used_hd_array();
  // device-use-note@-1 {{called by 'kernel'}}
  *out += device_used_hd_constexpr_array();
  // device-use-note@-1 {{called by 'kernel'}}
  *out += device_used_hd_lambda_array();
  // device-use-note@-1 {{called by 'kernel'}}
}
#endif
