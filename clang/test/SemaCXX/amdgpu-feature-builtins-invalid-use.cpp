// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fsyntax-only -verify=expected,spirv %s

bool predicate(bool x);
void pass_by_value(__amdgpu_feature_predicate_t x);

void invalid_uses(int *p, int x, const __amdgpu_feature_predicate_t &lv,
                  __amdgpu_feature_predicate_t &&rv) {
    __amdgpu_feature_predicate_t a;
    // expected-error@-1 {{'a' has type __amdgpu_feature_predicate_t, which is not constructible}}
    __amdgpu_feature_predicate_t b = __builtin_amdgcn_processor_is("gfx906");
    // expected-error@-1 {{'b' has type __amdgpu_feature_predicate_t, which is not constructible}}
    __amdgpu_feature_predicate_t c = lv;
    // expected-error@-1 {{'c' has type __amdgpu_feature_predicate_t, which is not constructible}}
    __amdgpu_feature_predicate_t d = rv;
    // expected-error@-1 {{'d' has type __amdgpu_feature_predicate_t, which is not constructible}}
    bool invalid_use_in_init_0 = __builtin_amdgcn_processor_is("gfx906");
    // expected-error@-1 {{'__builtin_amdgcn_processor_is("gfx906")' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided}}
    pass_by_value(__builtin_amdgcn_processor_is("gfx906"));
    // expected-error@-1 {{'x' has type __amdgpu_feature_predicate_t, which is not constructible}}
    bool invalid_use_in_init_1 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var);
    // expected-error@-1 {{'__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var)' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided}}
    if (bool invalid_use_in_init_2 = __builtin_amdgcn_processor_is("gfx906")) return;
    // expected-error@-1 {{'__builtin_amdgcn_processor_is("gfx906")' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided}}
    if (predicate(__builtin_amdgcn_processor_is("gfx1200"))) __builtin_amdgcn_s_sleep_var(x);
    // expected-error@-1 {{no matching function for call to 'predicate'}}
    // expected-error@-2 {{'__builtin_amdgcn_processor_is("gfx1200")' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided}}
    // spirv-error@-3 {{'__builtin_amdgcn_s_sleep_var' cannot be invoked in the current context, as it requires the 'gfx12-insts' feature(s)}}
}

void invalid_invocations(int x, const char* str) {
    if (__builtin_amdgcn_processor_is("not_an_amdgcn_gfx_id")) return;
    // expected-error@-1 {{the argument to __builtin_amdgcn_processor_is must be a valid AMDGCN processor identifier; 'not_an_amdgcn_gfx_id' is not valid}}
    // expected-note-re@-2 {{valid AMDGCN processor identifiers are: {{.*gfx.*}}}}
    if (__builtin_amdgcn_processor_is(str)) return;
    // expected-error@-1 {{the argument to __builtin_amdgcn_processor_is must be a string literal}}
    if (__builtin_amdgcn_is_invocable("__builtin_amdgcn_s_sleep_var")) return;
    // expected-error-re@-1 {{the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; {{.*__builtin_amdgcn_s_sleep_var.*}} is not valid}}
    else if (__builtin_amdgcn_is_invocable(str)) return;
    // expected-error-re@-1 {{the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; {{.*str.*}} is not valid}}
    else if (__builtin_amdgcn_is_invocable(x)) return;
    // expected-error-re@-1 {{the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; {{.*x.*}} is not valid}}
    else if (__builtin_amdgcn_is_invocable(__builtin_ia32_pause)) return;
    // expected-error@-1 {{use of undeclared identifier '__builtin_ia32_pause'}}
}

bool return_needs_cast() {
    return __builtin_amdgcn_processor_is("gfx900");
    // expected-error@-1 {{'__builtin_amdgcn_processor_is("gfx900")' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided}}
}
