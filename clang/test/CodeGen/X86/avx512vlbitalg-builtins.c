// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bitalg -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bitalg -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bitalg -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bitalg -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m256i test_mm256_popcnt_epi16(__m256i __A) {
  // CHECK-LABEL: test_mm256_popcnt_epi16
  // CHECK: @llvm.ctpop.v16i16
  return _mm256_popcnt_epi16(__A);
}
TEST_CONSTEXPR(match_v16hi(_mm256_popcnt_epi16((__m256i)(__v16hi){+5, -3, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129, +511, +1025}), 2, 15, 14, 1, 0, 8, 1, 9, 2, 2, 4, 2, 6, 2, 9, 2));

__m256i test_mm256_mask_popcnt_epi16(__m256i __A, __mmask16 __U, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_popcnt_epi16
  // CHECK: @llvm.ctpop.v16i16
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_popcnt_epi16(__A, __U, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_popcnt_epi16(_mm256_set1_epi16(-1), 0xF0F0, (__m256i)(__v16hi){+5, -3, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129, +511, +1025}), -1, -1, -1, -1, 0, 8, 1, 9, -1, -1, -1, -1, 6, 2, 9, 2));

__m256i test_mm256_maskz_popcnt_epi16(__mmask16 __U, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_popcnt_epi16
  // CHECK: @llvm.ctpop.v16i16
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_popcnt_epi16(__U, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_popcnt_epi16(0x0F0F, (__m256i)(__v16hi){+5, -3, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129, +511, +1025}), 2, 15, 14, 1, 0, 0, 0, 0, 2, 2, 4, 2, 0, 0, 0, 0));

__m128i test_mm_popcnt_epi16(__m128i __A) {
  // CHECK-LABEL: test_mm_popcnt_epi16
  // CHECK: @llvm.ctpop.v8i16
  return _mm_popcnt_epi16(__A);
}
TEST_CONSTEXPR(match_v8hi(_mm_popcnt_epi16((__m128i)(__v8hi){+5, -3, -10, +8, 0, -256, +256, -128}), 2, 15, 14, 1, 0, 8, 1, 9));

__m128i test_mm_mask_popcnt_epi16(__m128i __A, __mmask8 __U, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_popcnt_epi16
  // CHECK: @llvm.ctpop.v8i16
  // CHECK: select <8 x i1> %{{[0-9]+}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_popcnt_epi16(__A, __U, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_popcnt_epi16(_mm_set1_epi16(-1), 0xF0, (__m128i)(__v8hi){+5, -3, -10, +8, 0, -256, +256, -128}), -1, -1, -1, -1, 0, 8, 1, 9));

__m128i test_mm_maskz_popcnt_epi16(__mmask8 __U, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_popcnt_epi16
  // CHECK: @llvm.ctpop.v8i16
  // CHECK: select <8 x i1> %{{[0-9]+}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_popcnt_epi16(__U, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_popcnt_epi16(0x0F, (__m128i)(__v8hi){+5, -3, -10, +8, 0, -256, +256, -128}), 2, 15, 14, 1, 0, 0, 0, 0));

__m256i test_mm256_popcnt_epi8(__m256i __A) {
  // CHECK-LABEL: test_mm256_popcnt_epi8
  // CHECK: @llvm.ctpop.v32i8
  return _mm256_popcnt_epi8(__A);
}
TEST_CONSTEXPR(match_v32qi(_mm256_popcnt_epi8((__m256i)(__v32qi){+5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73, +5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73}), 2, 7, 6, 1, 0, 4, 1, 4, 2, 2, 4, 2, 6, 2, 4, 3, 2, 7, 6, 1, 0, 4, 1, 4, 2, 2, 4, 2, 6, 2, 4, 3));

__m256i test_mm256_mask_popcnt_epi8(__m256i __A, __mmask32 __U, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_popcnt_epi8
  // CHECK: @llvm.ctpop.v32i8
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_popcnt_epi8(__A, __U, __B);
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_popcnt_epi8(_mm256_set1_epi8(-1), 0xF00F, (__m256i)(__v32qi){+5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73, +5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73}), 2, 7, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 2, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));

__m256i test_mm256_maskz_popcnt_epi8(__mmask32 __U, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_popcnt_epi8
  // CHECK: @llvm.ctpop.v32i8
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_popcnt_epi8(__U, __B);
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_popcnt_epi8(0x0FF0, (__m256i)(__v32qi){+5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73, +5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73}), 0, 0, 0, 0, 0, 4, 1, 4, 2, 2, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_popcnt_epi8(__m128i __A) {
  // CHECK-LABEL: test_mm_popcnt_epi8
  // CHECK: @llvm.ctpop.v16i8
  return _mm_popcnt_epi8(__A);
}
TEST_CONSTEXPR(match_v16qi(_mm_popcnt_epi8((__m128i)(__v16qi){+5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73}), 2, 7, 6, 1, 0, 4, 1, 4, 2, 2, 4, 2, 6, 2, 4, 3));

__m128i test_mm_mask_popcnt_epi8(__m128i __A, __mmask16 __U, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_popcnt_epi8
  // CHECK: @llvm.ctpop.v16i8
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_popcnt_epi8(__A, __U, __B);
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_popcnt_epi8(_mm_set1_epi8(-1), 0xF00F, (__m128i)(__v16qi){+5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73}), 2, 7, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 2, 4, 3));

__m128i test_mm_maskz_popcnt_epi8(__mmask16 __U, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_popcnt_epi8
  // CHECK: @llvm.ctpop.v16i8
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_popcnt_epi8(__U, __B);
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_popcnt_epi8(0x0FF0, (__m128i)(__v16qi){+5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73}), 0, 0, 0, 0, 0, 4, 1, 4, 2, 2, 4, 2, 0, 0, 0, 0));

__mmask32 test_mm256_mask_bitshuffle_epi64_mask(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_bitshuffle_epi64_mask
  // CHECK: @llvm.x86.avx512.vpshufbitqmb.256
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_bitshuffle_epi64_mask(__U, __A, __B);
}

__mmask32 test_mm256_bitshuffle_epi64_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_bitshuffle_epi64_mask
  // CHECK: @llvm.x86.avx512.vpshufbitqmb.256
  return _mm256_bitshuffle_epi64_mask(__A, __B);
}

__mmask16 test_mm_mask_bitshuffle_epi64_mask(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_bitshuffle_epi64_mask
  // CHECK: @llvm.x86.avx512.vpshufbitqmb.128
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_bitshuffle_epi64_mask(__U, __A, __B);
}

__mmask16 test_mm_bitshuffle_epi64_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_bitshuffle_epi64_mask
  // CHECK: @llvm.x86.avx512.vpshufbitqmb.128
  return _mm_bitshuffle_epi64_mask(__A, __B);
}

