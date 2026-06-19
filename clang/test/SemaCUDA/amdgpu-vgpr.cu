// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 \
// RUN:   -fcuda-is-device -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__global__ void kernel() {
  int ok[4] __attribute__((amdgpu_vgpr)); // OK
  (void)ok;
}

__device__ void device_fn() {
  int bad __attribute__((amdgpu_vgpr)); // expected-error {{'amdgpu_vgpr' attribute can only be applied to local variables in '__global__' (kernel) functions}}
  (void)bad;
}

__host__ void host_fn() {
  int bad __attribute__((amdgpu_vgpr)); // expected-error {{'amdgpu_vgpr' attribute can only be applied to local variables in '__global__' (kernel) functions}}
  (void)bad;
}

// Not a local variable.
int global_var __attribute__((amdgpu_vgpr)); // expected-error {{'amdgpu_vgpr' attribute only applies to local variables}}

__global__ void takes_no_args() {
  // Attribute does not accept arguments.
  int bad __attribute__((amdgpu_vgpr(1))); // expected-error {{'amdgpu_vgpr' attribute takes no arguments}}
  (void)bad;
}
