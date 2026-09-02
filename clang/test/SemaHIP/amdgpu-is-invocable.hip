// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple amdgcn -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple spirv64-amd-amdhsa -Wno-unused-value %s

// expected-no-diagnostics

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ void foo() {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_permlanex16))
        return __builtin_trap();
}

__global__ void bar() {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_permlanex16))
        return __builtin_trap();
}
