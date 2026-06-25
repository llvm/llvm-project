// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 \
// RUN:   -fcuda-is-device -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__global__ void kernel() {
  int ok[4] __attribute__((amdgpu_vgpr)); // OK
  (void)ok;
}

__global__ void initialized() {
  // Register-backed storage has undefined initial contents, so (like
  // __shared__) it cannot be initialized.
  int bad __attribute__((amdgpu_vgpr)) = 7; // expected-error {{a variable with the 'amdgpu_vgpr' attribute cannot have an initializer}}
  int arr[2] __attribute__((amdgpu_vgpr)) = {1, 2}; // expected-error {{a variable with the 'amdgpu_vgpr' attribute cannot have an initializer}}
  (void)bad;
  (void)arr;
}

__device__ void device_fn() {
  // Allowed in device functions too (like __shared__); the backend handles
  // references to the global from non-kernel functions.
  int ok __attribute__((amdgpu_vgpr)); // OK
  (void)ok;
}

__host__ void host_fn() {
  int bad __attribute__((amdgpu_vgpr)); // expected-error {{'amdgpu_vgpr' variables are not allowed in __host__ functions}}
  (void)bad;
}

// Not a local variable.
int global_var __attribute__((amdgpu_vgpr)); // expected-error {{'amdgpu_vgpr' attribute only applies to local variables}}

__global__ void takes_no_args() {
  // Attribute does not accept arguments.
  int bad __attribute__((amdgpu_vgpr(1))); // expected-error {{'amdgpu_vgpr' attribute takes no arguments}}
  (void)bad;
}

__global__ void bad_storage(int n) {
  // A static-storage local is not a LocalVar subject; a VLA is rejected as not
  // fixed-size. Both must avoid silently ignoring the attribute.
  static int s __attribute__((amdgpu_vgpr)); // expected-error {{'amdgpu_vgpr' attribute only applies to local variables}}
  int vla[n] __attribute__((amdgpu_vgpr));   // expected-error {{the 'amdgpu_vgpr' attribute requires an automatic, fixed-size local variable}}
  (void)s;
  (void)vla;
}
