// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm-only -verify %s

// Test import on non-wasm-variable (addrspace 0)
extern const int defined_g_addrspace0
    __attribute__((import_module("js"))); // expected-error {{import attribute cannot be applied to a non-wasm-variable global}}

int get_val(void) { return defined_g_addrspace0; }
