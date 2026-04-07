// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64-amd-amdhsa -Wamdgpu-unguarded-builtin-usage %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ void g();

__device__ void f(int x, bool b) {
    long v15_16;
    __asm volatile("v_lshlrev_b64 v[15:16], 0, %0" : "={v[15:16]}"(v15_16) : "v"(x)); // expected-warning {{the 'v_lshlrev_b64 v[15:16], 0, $0' ASM sequence might be invalid for some AMDGPU targets}}
    // expected-note@-1 {{enclose the 'v_lshlrev_b64 v[15:16], 0, $0' ASM sequence in a scope controlled by a __builtin_amdgcn_is_processor check to silence this warning}}

    if (__builtin_amdgcn_processor_is("gfx90a")) {
        long v15_16;
        __asm volatile("v_lshlrev_b64 v[15:16], 0, %0" : "={v[15:16]}"(v15_16) : "v"(x));
    }

    if (!__builtin_amdgcn_processor_is("gfx90a")) {
        long v15_16;
        __asm volatile("v_lshlrev_b64 v[15:16], 0, %0" : "={v[15:16]}"(v15_16) : "v"(x)); // expected-warning {{the 'v_lshlrev_b64 v[15:16], 0, $0' ASM sequence might be invalid for some AMDGPU targets}}
        // expected-note@-1 {{enclose the 'v_lshlrev_b64 v[15:16], 0, $0' ASM sequence in a scope controlled by a __builtin_amdgcn_is_processor check to silence this warning}}
    }

    __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var) ? __builtin_amdgcn_s_sleep_var(x) : __builtin_trap();
    !__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var) ? __builtin_amdgcn_s_sleep_var(x) : __builtin_trap(); // expected-error {{'__builtin_amdgcn_s_sleep_var' cannot be invoked in the current context, as it requires the 'gfx12-insts' feature(s)}}

    const auto lambda = [=] __device__  () {
        __builtin_amdgcn_s_sleep(42); // expected-warning {{'__builtin_amdgcn_s_sleep' might be unavailable on some AMDGPU targets}}
        // expected-note@-1 {{enclose '__builtin_amdgcn_s_sleep' in a __builtin_amdgcn_is_invocable check to silence this warning}}

        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var))
            __builtin_amdgcn_s_sleep_var(x);
    };

    const auto generic_lambda = [] __device__ (auto&& y) {
        __builtin_amdgcn_s_sleep(42); // expected-warning {{'__builtin_amdgcn_s_sleep' might be unavailable on some AMDGPU targets}}
        // expected-note@-1 {{enclose '__builtin_amdgcn_s_sleep' in a __builtin_amdgcn_is_invocable check to silence this warning}}

        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var)) {
            __builtin_amdgcn_s_sleep_var(y);
            // Has the same requirements - gfx12-insts, thus correct, but we should still warn.
            __builtin_amdgcn_s_barrier_signal_var(nullptr, y); // expected-warning {{'__builtin_amdgcn_s_barrier_signal_var' might be unavailable on some AMDGPU targets}}
            // expected-note@-1 {{enclose '__builtin_amdgcn_s_barrier_signal_var' in a __builtin_amdgcn_is_invocable check to silence this warning}}
        }
    };

    __builtin_amdgcn_s_sleep(42); // expected-warning {{'__builtin_amdgcn_s_sleep' might be unavailable on some AMDGPU targets}}
    // expected-note@-1 {{enclose '__builtin_amdgcn_s_sleep' in a __builtin_amdgcn_is_invocable check to silence this warning}}

    if (__builtin_amdgcn_processor_is("gfx1201")) {
        if (__builtin_amdgcn_processor_is("gfx906")) // expected-error {{conflicting check for AMDGCN processor '__builtin_amdgcn_processor_is("gfx906")' found in a scope already controlled by a check for AMDGCN processor}}
        // expected-note@-2 {{predicate guard, with establishes the context, inserted here}}
            __builtin_trap();
    }

    if (__builtin_amdgcn_processor_is("gfx900")) {
        if (__builtin_amdgcn_processor_is("gfx900")) // This is fine, albeit potentially spurious.
            ++x;
    }

    if (__builtin_amdgcn_processor_is("gfx1030"))
        __builtin_amdgcn_s_barrier_signal_isfirst(42); // expected-error {{'__builtin_amdgcn_s_barrier_signal_isfirst' cannot be invoked in the current context, as it requires the 'gfx12-insts' feature(s), which 'gfx1030' does not provide}}
        // expected-note@-2 {{predicate guard, with establishes the context, inserted here}}

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

__attribute__((target("arch=gfx1030")))
__device__ void h(int x) {
    if (__builtin_amdgcn_processor_is("gfx1030")) // Fine, same processor
        return;

    long v15_16;
    __asm volatile("v_lshlrev_b64 v[15:16], 0, %0" : "={v[15:16]}"(v15_16) : "v"(x)); // "Fine", explicit gfx target

    __builtin_amdgcn_s_ttracedata_imm(42); // expected-warning {{'__builtin_amdgcn_s_ttracedata_imm' might be unavailable on some AMDGPU targets}}
    // expected-note@-1 {{enclose '__builtin_amdgcn_s_ttracedata_imm' in a __builtin_amdgcn_is_invocable check to silence this warning}}

    __builtin_amdgcn_s_barrier_signal_isfirst(42); // expected-error {{'__builtin_amdgcn_s_barrier_signal_isfirst' cannot be invoked in the current context, as it requires the 'gfx12-insts' feature(s), which 'gfx1030' does not provide}}
    // expected-note@-12 {{predicate guard, with establishes the context, inserted here}}

    if (__builtin_amdgcn_processor_is("gfx906")) // expected-error {{conflicting check for AMDGCN processor '__builtin_amdgcn_processor_is("gfx906")' found in a scope already controlled by a check for AMDGCN processor}}
    // expected-note@-15 {{predicate guard, with establishes the context, inserted here}}
        __builtin_trap();
}

__attribute__((target("gfx11-insts")))
__device__ void i(int x) {
    __builtin_amdgcn_s_wait_event_export_ready(); // expected-warning {{'__builtin_amdgcn_s_wait_event_export_ready' might be unavailable on some AMDGPU targets}}
    // expected-note@-1 {{enclose '__builtin_amdgcn_s_wait_event_export_ready' in a __builtin_amdgcn_is_invocable check to silence this warning}}

    __builtin_amdgcn_s_barrier_signal_isfirst(42); // expected-error {{'__builtin_amdgcn_s_barrier_signal_isfirst' cannot be invoked in the current context, as it requires the 'gfx12-insts' feature(s)}}
}

__attribute__((target("gfx11-insts,gfx12-insts")))
__device__ void j(int x) {
    __builtin_amdgcn_s_wait_event_export_ready(); // expected-warning {{'__builtin_amdgcn_s_wait_event_export_ready' might be unavailable on some AMDGPU targets}}
    // expected-note@-1 {{enclose '__builtin_amdgcn_s_wait_event_export_ready' in a __builtin_amdgcn_is_invocable check to silence this warning}}

    __builtin_amdgcn_s_barrier_signal_isfirst(42); // expected-warning {{'__builtin_amdgcn_s_barrier_signal_isfirst' might be unavailable on some AMDGPU targets}}
    // expected-note@-1 {{enclose '__builtin_amdgcn_s_barrier_signal_isfirst' in a __builtin_amdgcn_is_invocable check to silence this warning}}
}
