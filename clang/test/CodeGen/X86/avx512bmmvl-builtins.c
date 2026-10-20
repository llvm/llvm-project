// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bmm -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bmm -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bmm -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bmm -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bmm -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bmm -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bmm -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bmm -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m256i test_mm256_bmacor16x16x16(__m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: test_mm256_bmacor16x16x16
  // CHECK: @llvm.x86.avx512.vbmacor.v16hi
  return _mm256_bmacor16x16x16(__A, __B, __C);
}
// All-ones * all-ones with OR reduction sets every result bit (C = 0).
TEST_CONSTEXPR(match_v16hi(_mm256_bmacor16x16x16(_mm256_set1_epi16(-1), _mm256_set1_epi16(-1), _mm256_setzero_si256()),
                           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
// A == 0 yields a zero product, so the accumulator passes through unchanged.
TEST_CONSTEXPR(match_v16hi(_mm256_bmacor16x16x16(_mm256_setzero_si256(), _mm256_set1_epi16(-1), _mm256_set1_epi16(0x1234)),
                           0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234,
                           0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234, 0x1234));

__m256i test_mm256_bmacxor16x16x16(__m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: test_mm256_bmacxor16x16x16
  // CHECK: @llvm.x86.avx512.vbmacxor.v16hi
  return _mm256_bmacxor16x16x16(__A, __B, __C);
}
// All-ones * all-ones with XOR reduction: 16 product terms per bit cancel to 0.
TEST_CONSTEXPR(match_v16hi(_mm256_bmacxor16x16x16(_mm256_set1_epi16(-1), _mm256_set1_epi16(-1), _mm256_setzero_si256()),
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm128_bitrev_epi8(__m128i __A) {
  // CHECK-LABEL: test_mm128_bitrev_epi8
  // CHECK: @llvm.bitreverse.v16i8
  return _mm128_bitrev_epi8(__A);
}
TEST_CONSTEXPR(match_v16qi(_mm128_bitrev_epi8((__m128i)(__v16qi){
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
    0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA}),
    (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
    0x00, (char)0xFF, (char)0xF0, 0x0F, (char)0xCC, 0x33, (char)0xAA, 0x55));

__m256i test_mm256_bitrev_epi8(__m256i __A) {
  // CHECK-LABEL: test_mm256_bitrev_epi8
  // CHECK: @llvm.bitreverse.v32i8
  return _mm256_bitrev_epi8(__A);
}
TEST_CONSTEXPR(match_v32qi(_mm256_bitrev_epi8((__m256i)(__v32qi){
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
    0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
    0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA}),
    (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
    0x00, (char)0xFF, (char)0xF0, 0x0F, (char)0xCC, 0x33, (char)0xAA, 0x55,
    (char)0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
    0x00, (char)0xFF, (char)0xF0, 0x0F, (char)0xCC, 0x33, (char)0xAA, 0x55));

__m128i test_mm128_mask_bitrev_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm128_mask_bitrev_epi8
  // CHECK: @llvm.bitreverse.v16i8
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm128_mask_bitrev_epi8(__U, __A, __B);
}
// Mask 0x5555: even bytes take the bit-reversed value, odd bytes pass __B through.
TEST_CONSTEXPR(match_v16qi(_mm128_mask_bitrev_epi8((__mmask16)0x5555,
    (__m128i)(__v16qi){0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
                       0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA},
    (__m128i)(__v16qi){(char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99,
                       (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99}),
    (char)0x80, (char)0x99, 0x20, (char)0x99, 0x08, (char)0x99, 0x02, (char)0x99,
    0x00, (char)0x99, (char)0xF0, (char)0x99, (char)0xCC, (char)0x99, (char)0xAA, (char)0x99));

__m128i test_mm128_maskz_bitrev_epi8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm128_maskz_bitrev_epi8
  // CHECK: @llvm.bitreverse.v16i8
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm128_maskz_bitrev_epi8(__U, __A);
}
// Mask 0x5555: even bytes take the bit-reversed value, odd bytes are zeroed.
TEST_CONSTEXPR(match_v16qi(_mm128_maskz_bitrev_epi8((__mmask16)0x5555,
    (__m128i)(__v16qi){0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
                       0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA}),
    (char)0x80, 0x00, 0x20, 0x00, 0x08, 0x00, 0x02, 0x00,
    0x00, 0x00, (char)0xF0, 0x00, (char)0xCC, 0x00, (char)0xAA, 0x00));

__m256i test_mm256_mask_bitrev_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_bitrev_epi8
  // CHECK: @llvm.bitreverse.v32i8
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_bitrev_epi8(__U, __A, __B);
}
// Mask 0x55555555: even bytes take the bit-reversed value, odd bytes pass __B through.
TEST_CONSTEXPR(match_v32qi(_mm256_mask_bitrev_epi8((__mmask32)0x55555555,
    (__m256i)(__v32qi){0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
                       0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA,
                       0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
                       0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA},
    (__m256i)(__v32qi){(char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99,
                       (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99,
                       (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99,
                       (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99, (char)0x99}),
    (char)0x80, (char)0x99, 0x20, (char)0x99, 0x08, (char)0x99, 0x02, (char)0x99,
    0x00, (char)0x99, (char)0xF0, (char)0x99, (char)0xCC, (char)0x99, (char)0xAA, (char)0x99,
    (char)0x80, (char)0x99, 0x20, (char)0x99, 0x08, (char)0x99, 0x02, (char)0x99,
    0x00, (char)0x99, (char)0xF0, (char)0x99, (char)0xCC, (char)0x99, (char)0xAA, (char)0x99));

__m256i test_mm256_maskz_bitrev_epi8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_bitrev_epi8
  // CHECK: @llvm.bitreverse.v32i8
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_bitrev_epi8(__U, __A);
}
// Mask 0x55555555: even bytes take the bit-reversed value, odd bytes are zeroed.
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_bitrev_epi8((__mmask32)0x55555555,
    (__m256i)(__v32qi){0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
                       0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA,
                       0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
                       0x00, (char)0xFF, 0x0F, (char)0xF0, 0x33, (char)0xCC, 0x55, (char)0xAA}),
    (char)0x80, 0x00, 0x20, 0x00, 0x08, 0x00, 0x02, 0x00,
    0x00, 0x00, (char)0xF0, 0x00, (char)0xCC, 0x00, (char)0xAA, 0x00,
    (char)0x80, 0x00, 0x20, 0x00, 0x08, 0x00, 0x02, 0x00,
    0x00, 0x00, (char)0xF0, 0x00, (char)0xCC, 0x00, (char)0xAA, 0x00));
