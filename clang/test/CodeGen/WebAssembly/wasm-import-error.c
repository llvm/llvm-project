// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm-only -verify=fun -DTEST_FUN %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm-only -Wno-extern-initializer -verify=var -DTEST_VAR %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm-only -verify=addrspace -DTEST_ADDRSPACE %s

#ifdef TEST_FUN
void defined_fn(void) __attribute__((import_module("js"))) {} // fun-error {{import attribute cannot be applied to a definition}}
#endif

#ifdef TEST_VAR
// Test definition inline
extern const int __attribute__((address_space(1))) defined_g_inline
    __attribute__((import_module("js"))) = 42; // var-error {{import attribute cannot be applied to a definition}}
#endif

#ifdef TEST_ADDRSPACE
// Test import on non-wasm-variable (addrspace 0)
extern const int defined_g_addrspace0
    __attribute__((import_module("js"))); // addrspace-error {{import attribute cannot be applied to a non-wasm-variable global}}

int get_val(void) { return defined_g_addrspace0; }
#endif
