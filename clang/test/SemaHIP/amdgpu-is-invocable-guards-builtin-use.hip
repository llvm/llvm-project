// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64-amd-amdhsa -Wamdgpu-unguarded-builtin-usage %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ void g();

__device__ void f(int x, bool b) {
    const auto lambda = [=] __device__  () {
        __builtin_amdgcn_s_sleep(42); // expected-warning {{'__builtin_amdgcn_s_sleep' might be unavailable on some AMDGPU targets}}
        // expected-note@-1 {{enclose '__builtin_amdgcn_s_sleep' in a __builtin_amdgcn_is_invocable check to silence this warning}}

        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var))
            __builtin_amdgcn_s_sleep_var(x);
    };

    const auto generic_lambda = [] __device__ (auto&& y) {
        __builtin_amdgcn_s_sleep(42); // expected-warning {{'__builtin_amdgcn_s_sleep' might be unavailable on some AMDGPU targets}}
        // expected-note@-1 {{enclose '__builtin_amdgcn_s_sleep' in a __builtin_amdgcn_is_invocable check to silence this warning}}

        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var))
            __builtin_amdgcn_s_sleep_var(y);
    };

    __builtin_amdgcn_s_sleep(42); // expected-warning {{'__builtin_amdgcn_s_sleep' might be unavailable on some AMDGPU targets}}
    // expected-note@-1 {{enclose '__builtin_amdgcn_s_sleep' in a __builtin_amdgcn_is_invocable check to silence this warning}}

    // processor_is does not (yet) guard
    if (__builtin_amdgcn_processor_is("gfx900"))
        __builtin_amdgcn_s_sleep_var(x); // expected-warning {{'__builtin_amdgcn_s_sleep_var' might be unavailable on some AMDGPU targets}}
        // expected-note@-1 {{enclose '__builtin_amdgcn_s_sleep_var' in a __builtin_amdgcn_is_invocable check to silence this warning}}

    // Direct guard
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep))
        __builtin_amdgcn_s_sleep(42);

    // Guarded scope
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var)) {
        if (b) {
            g();
            while (--x > 42)
                __builtin_amdgcn_s_sleep_var(x);
        }
    }
}
