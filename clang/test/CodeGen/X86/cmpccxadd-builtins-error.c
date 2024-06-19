// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +cmpccxadd  -fsyntax-only -verify

#include <immintrin.h>

int test_cmpccxadd32(void *__A, int __B, int __C) {
  return _cmpccxadd_epi32(__A, __B, __C, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
}

long long test_cmpccxadd64(void *__A, long long __B, long long __C) {
  return _cmpccxadd_epi64(__A, __B, __C, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
}
