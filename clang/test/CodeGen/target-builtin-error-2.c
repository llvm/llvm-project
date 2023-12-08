// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o -
#define __MM_MALLOC_H

#include <x86intrin.h>

// Since we do code generation on a function level this needs to error out since
// the subtarget feature won't be available.
__m128 wombat(__m128i a) {
  if (__builtin_cpu_supports("avx"))
    return __builtin_ia32_vpermilvarps((__v4sf) {0.0f, 1.0f, 2.0f, 3.0f}, (__v4si)a); // expected-error {{'__builtin_ia32_vpermilvarps' needs target feature avx}}
  else
    return (__m128){0, 0};
}
