// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm-only -Wno-extern-initializer -verify %s

// Test definition inline
extern const int __attribute__((address_space(1))) defined_g_inline
    __attribute__((import_module("js"))) = 42; // expected-error {{import attribute cannot be applied to a definition}}
