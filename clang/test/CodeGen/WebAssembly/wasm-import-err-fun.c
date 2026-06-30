// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm-only -verify %s

void defined_fn(void) __attribute__((import_module("js"))) {} // expected-error {{import attribute cannot be applied to a definition}}
