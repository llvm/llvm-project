// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +prefetchi  -fsyntax-only -verify

#include <immintrin.h>

void test_invalid_prefetchi(void* p) {
  __builtin_ia32_prefetchi(p, 1); // expected-error {{argument value 1 is outside the valid range [2, 3]}}
}
