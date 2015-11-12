// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o -
#define __MM_MALLOC_H
#include <x86intrin.h>

int baz(__m256i a) {
  return _mm256_extract_epi32(a, 3); // expected-error {{function 'baz' and always_inline callee function '_mm256_extract_epi32' are required to have matching target features}}
}
