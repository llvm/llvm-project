// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o -
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128 foo(__m128 a, __m128 b) {
  return __builtin_ia32_addsubps(b, a); // expected-error {{'__builtin_ia32_addsubps' needs target feature sse3}}
}
