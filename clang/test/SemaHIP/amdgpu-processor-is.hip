// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple amdgcn -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple spirv64-amd-amdhsa -Wno-unused-value %s

// The queried processor is accepted whenever it is a recognized AMDGCN
// processor, independent of the target's own subarch. A processor that does not
// match the target simply evaluates to false, so none of these error.
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgpu9.00-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgpu9-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgpu9.02-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgpu10.10-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple amdgpu10.10-amd-amdhsa -Wno-unused-value %s

// expected-no-diagnostics

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ void foo() {
    if (__builtin_amdgcn_processor_is("gfx900"))
        return __builtin_trap();
    if (__builtin_amdgcn_processor_is("gfx1010"))
        return __builtin_trap();
}

__global__ void bar() {
    if (__builtin_amdgcn_processor_is("gfx900"))
        return __builtin_trap();
}
