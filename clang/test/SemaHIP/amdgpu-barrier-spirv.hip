// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 %s -fsyntax-only -fcuda-is-device -std=c++17 -triple spirv64-amd-amdhsa -verify

// TODO: Make this an error

#define __device__ __attribute__((device))

__amdgpu_named_workgroup_barrier_t bar; // expected-error{{AMDGPU built-in type '__amdgpu_named_workgroup_barrier_t' is not supported on target 'spirv64-amd-amdhsa'}}

__amdgpu_named_workgroup_barrier_t *getBar() { // expected-error{{AMDGPU built-in type '__amdgpu_named_workgroup_barrier_t *' is not supported on target 'spirv64-amd-amdhsa'}}
  return &bar;
}

void foo(__amdgpu_named_workgroup_barrier_t x) { // expected-error{{AMDGPU built-in type '__amdgpu_named_workgroup_barrier_t' is not supported on target 'spirv64-amd-amdhsa'}}
  __amdgpu_named_workgroup_barrier_t v; // expected-error{{AMDGPU built-in type '__amdgpu_named_workgroup_barrier_t' is not supported on target 'spirv64-amd-amdhsa'}}
}
