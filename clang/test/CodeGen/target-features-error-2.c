// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o -
#define __MM_MALLOC_H
#include <x86intrin.h>

int baz(__m256i a) {
  return _mm256_extract_epi32(a, 3); // expected-error {{always_inline function '_mm256_extract_epi32' requires target feature 'sse4.2', but would be inlined into function 'baz' that is compiled without support for 'sse4.2'}}
}
