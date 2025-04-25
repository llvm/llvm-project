// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm %s -o - 2>&1 | FileCheck %s

bool predicate(bool x) { return x; }

void invalid_uses(int* p, int x, bool (*pfn)(bool)) {
    // CHECK: error: cannot initialize a variable of type 'bool' with an rvalue of type 'void'
    bool invalid_use_in_init_0 = __builtin_amdgcn_processor_is("gfx906");
    // CHECK: error: cannot initialize a variable of type 'const bool' with an rvalue of type 'void'
    const bool invalid_use_in_init_1 = !__builtin_amdgcn_processor_is("gfx906");
    // CHECK: error: cannot initialize a variable of type 'bool' with an rvalue of type 'void'
    bool invalid_use_in_init_2 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var);
    // CHECK: error: cannot initialize a variable of type 'bool' with an rvalue of type 'void'
    bool invalid_use_in_init_3 = !__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var);
    // CHECK: error: variable has incomplete type 'const void'
    const auto invalid_use_in_init_4 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_wait_event_export_ready) || __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var);
    // CHECK: error: variable has incomplete type 'const void'
    const auto invalid_use_in_init_5 = __builtin_amdgcn_processor_is("gfx906") || __builtin_amdgcn_processor_is("gfx900");
    // CHECK: error: variable has incomplete type 'const void'
    const auto invalid_use_in_init_6 = __builtin_amdgcn_processor_is("gfx906") || __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep);
    // CHECK: error: value of type 'void' is not contextually convertible to 'bool'
    __builtin_amdgcn_processor_is("gfx1201")
        ? __builtin_amdgcn_s_sleep_var(x) : __builtin_amdgcn_s_sleep(42);
    // CHECK: error: no matching function for call to 'predicate'
    if (predicate(__builtin_amdgcn_processor_is("gfx1200"))) __builtin_amdgcn_s_sleep_var(x);
    // CHECK: note: candidate function not viable: cannot convert argument of incomplete type 'void' to 'bool' for 1st argument
}

void invalid_invocations(int x, const char* str) {
    // CHECK: error: the argument to __builtin_amdgcn_processor_is must be a valid AMDGCN processor identifier; 'not_an_amdgcn_gfx_id' is not valid
    if (__builtin_amdgcn_processor_is("not_an_amdgcn_gfx_id")) return;
    // CHECK: error: the argument to __builtin_amdgcn_processor_is must be a string literal
    if (__builtin_amdgcn_processor_is(str)) return;

    // CHECK: error: the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; `'"__builtin_amdgcn_s_sleep_var"'` is not valid
    if (__builtin_amdgcn_is_invocable("__builtin_amdgcn_s_sleep_var")) return;
    // CHECK: error: the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; `'str'` is not valid
    else if (__builtin_amdgcn_is_invocable(str)) return;
    // CHECK: error: the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; `'x'` is not valid
    else if (__builtin_amdgcn_is_invocable(x)) return;
    // CHECK: error: use of undeclared identifier '__builtin_ia32_pause'
    else if (__builtin_amdgcn_is_invocable(__builtin_ia32_pause)) return;
}
