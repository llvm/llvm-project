// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm %s -o - 2>&1 | FileCheck %s

bool predicate(bool x);
void pass_by_value(__amdgpu_feature_predicate_t x);

void invalid_uses(int *p, int x, const __amdgpu_feature_predicate_t &lv,
                  __amdgpu_feature_predicate_t &&rv) {
    // CHECK: error: 'a' has type __amdgpu_feature_predicate_t, which is not constructible
    __amdgpu_feature_predicate_t a;
    // CHECK: error: 'b' has type __amdgpu_feature_predicate_t, which is not constructible
    __amdgpu_feature_predicate_t b = __builtin_amdgcn_processor_is("gfx906");
    // CHECK: error: 'c' has type __amdgpu_feature_predicate_t, which is not constructible
    __amdgpu_feature_predicate_t c = lv;
    // CHECK: error: 'd' has type __amdgpu_feature_predicate_t, which is not constructible
    __amdgpu_feature_predicate_t d = rv;
    // CHECK: error: '__builtin_amdgcn_processor_is("gfx906")' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided
    bool invalid_use_in_init_0 = __builtin_amdgcn_processor_is("gfx906");
    // CHECK: error: 'x' has type __amdgpu_feature_predicate_t, which is not constructible
    pass_by_value(__builtin_amdgcn_processor_is("gfx906"));
    // CHECK: error: '__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var)' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided
    bool invalid_use_in_init_1 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep_var);
    // CHECK: error: '__builtin_amdgcn_processor_is("gfx906")' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided
    if (bool invalid_use_in_init_2 = __builtin_amdgcn_processor_is("gfx906")) return;
    // CHECK: error: '__builtin_amdgcn_processor_is("gfx1200")' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided
    if (predicate(__builtin_amdgcn_processor_is("gfx1200"))) __builtin_amdgcn_s_sleep_var(x);
}

void invalid_invocations(int x, const char* str) {
    // CHECK: error: the argument to __builtin_amdgcn_processor_is must be a valid AMDGCN processor identifier; 'not_an_amdgcn_gfx_id' is not valid
    // CHECK-DAG: note: valid AMDGCN processor identifiers are: {{.*}}gfx{{.*}}
    if (__builtin_amdgcn_processor_is("not_an_amdgcn_gfx_id")) return;
    // CHECK: error: the argument to __builtin_amdgcn_processor_is must be a string literal
    if (__builtin_amdgcn_processor_is(str)) return;
    // CHECK: error: the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; {{.*}}__builtin_amdgcn_s_sleep_var{{.*}} is not valid
    if (__builtin_amdgcn_is_invocable("__builtin_amdgcn_s_sleep_var")) return;
    // CHECK: error: the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; {{.*}}str{{.*}} is not valid
    else if (__builtin_amdgcn_is_invocable(str)) return;
    // CHECK: error: the argument to __builtin_amdgcn_is_invocable must be either a target agnostic builtin or an AMDGCN target specific builtin; {{.*}}x{{.*}} is not valid
    else if (__builtin_amdgcn_is_invocable(x)) return;
    // CHECK: error: use of undeclared identifier '__builtin_ia32_pause'
    else if (__builtin_amdgcn_is_invocable(__builtin_ia32_pause)) return;
}

bool return_needs_cast() {
    // CHECK: error: '__builtin_amdgcn_processor_is("gfx900")' must be explicitly cast to 'bool'; however, please note that this is almost always an error and that it prevents the effective guarding of target dependent code, and thus should be avoided
    return __builtin_amdgcn_processor_is("gfx900");
}
