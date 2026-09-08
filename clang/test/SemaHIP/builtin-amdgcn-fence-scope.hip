// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple amdgpu7.00-amd-amdhsa -fcuda-is-device -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-unknown-linux-gnu -aux-triple amdgpu7.00-amd-amdhsa -verify %s

// Verify the synchronization scope name is validated against the AMDGPU target
// on both the device compilation and host.

#define __device__ __attribute__((device))

__device__ void valid_scopes() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}

__device__ void invalid_scope() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "foobar"); // expected-error {{unsupported atomic synchronization scope 'foobar'}}
}
