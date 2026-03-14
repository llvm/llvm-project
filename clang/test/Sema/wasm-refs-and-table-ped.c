// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=expected -triple wasm32 -Wno-unused-value -target-feature +reference-types %s
// No error should be emitted.
static __externref_t table[0]; // expected-no-diagnostics
