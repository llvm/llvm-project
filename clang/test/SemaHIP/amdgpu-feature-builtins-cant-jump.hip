// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn -target-cpu gfx900 -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn -target-cpu gfx1201 -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple amdgcn -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple spirv64-amd-amdhsa -Wno-unused-value %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ void f(int *ptr, int size, bool f) {
    int i = 0;
    if (f)
        goto label; // expected-error {{cannot jump from this goto statement to its label}}

    if (__builtin_amdgcn_processor_is("gfx900")) { // expected-note {{jump enters statement controlled by AMDGPU feature predicate}}
        for (i = 0; i < size; ++i) {
            label:
            ptr[i] = i;
        }
    }
}

__device__ void g(int *ptr, int size, bool f) {
    int i = 0;
    if (f)
        goto label; // expected-error {{cannot jump from this goto statement to its label}}

    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var)) { // expected-note {{jump enters statement controlled by AMDGPU feature predicate}}
        for (i = 0; i < size; ++i) {
            label:
            ptr[i] = i;
        }
    }
}

__global__ void h(int *ptr, int size, bool f) {
    int i = 0;
    if (f)
        goto label; // expected-error {{cannot jump from this goto statement to its label}}

    if (__builtin_amdgcn_processor_is("gfx900")) { // expected-note {{jump enters statement controlled by AMDGPU feature predicate}}
        for (i = 0; i < size; ++i) {
            label:
            ptr[i] = i;
        }
    }
}

__global__ void i(int *ptr, int size, bool f) {
    int i = 0;
    if (f)
        goto label; // expected-error {{cannot jump from this goto statement to its label}}

    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var)) { // expected-note {{jump enters statement controlled by AMDGPU feature predicate}}
        for (i = 0; i < size; ++i) {
            label:
            ptr[i] = i;
        }
    }
}
