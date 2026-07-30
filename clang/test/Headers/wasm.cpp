// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +simd128 -fsyntax-only -ffreestanding %s -verify
// expected-no-diagnostics

#if defined(__wasm__) && defined(__wasm_simd128__)

extern "C++" {
#include <wasm_simd128.h>
}

#endif
