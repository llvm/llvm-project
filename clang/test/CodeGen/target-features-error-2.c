// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o - -D NEED_AVX_1
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o - -D NEED_AVX_2
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o - -D NEED_AVX512f
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +movdir64b -S -verify -o - -D NEED_MOVDIRI
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +avx512vnni -target-feature +movdiri -S -verify -o - -D NEED_CLWB

#define __MM_MALLOC_H
#include <x86intrin.h>

#if NEED_AVX_1
int baz(__m256i a) {
  return _mm256_extract_epi32(a, 3); // expected-error {{'__builtin_ia32_vec_ext_v8si' needs target feature avx}}
}
#endif

#if NEED_AVX_2
__m128 need_avx(__m128 a, __m128 b) {
  return _mm_cmp_ps(a, b, 8); // expected-error {{'__builtin_ia32_cmpps' needs target feature avx}}
}
#endif

#if NEED_AVX512f
unsigned short need_avx512f(unsigned short a, unsigned short b) {
  return __builtin_ia32_korhi(a, b); // expected-error {{'__builtin_ia32_korhi' needs target feature avx512f}}
}
#endif

#if NEED_MOVDIRI
void need_movdiri(unsigned int *a, unsigned int b) {
  __builtin_ia32_directstore_u32(a, b); // expected-error {{'__builtin_ia32_directstore_u32' needs target feature movdiri}}
}
#endif

#if NEED_CLWB
static __inline__ void
 __attribute__((__always_inline__, __nodebug__,  __target__("avx512vnni,clwb,movdiri,movdir64b")))
 func(unsigned int *a, unsigned int b)
{
  __builtin_ia32_directstore_u32(a, b);
}

void need_clwb(unsigned int *a, unsigned int b) {
  func(a, b); // expected-error {{always_inline function 'func' requires target feature 'clwb', but would be inlined into function 'need_clwb' that is compiled without support for 'clwb'}}

}
#endif
