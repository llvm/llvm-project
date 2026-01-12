// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512vbmi \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s

#include <immintrin.h>
__m512i test_multishift(__m512i x, __m512i y) {
  return _mm512_multishift_epi64_epi8(x, y);
}

// // CHECK: cir.func @test_multishift
// // CHECK: cir.call @__builtin_ia32_vpmultishiftqb512

__m128i test_mm_multishift_epi64_epi8(__m128i __X, __m128i __Y)
{
  return (__m128i)_mm_multishift_epi64_epi8((__v16qi)__X, (__v16qi)__Y);
}



__m256i test_mm256_multishift_epi64_epi8(__m256i __X, __m256i __Y)
{
  return (__m256i)_mm256_multishift_epi64_epi8((__v32qi)__X, (__v32qi)__Y);
}