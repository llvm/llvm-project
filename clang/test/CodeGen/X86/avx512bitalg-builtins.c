// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bitalg -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bitalg -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bitalg -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bitalg -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m512i test_mm512_popcnt_epi16(__m512i __A) {
  // CHECK-LABEL: test_mm512_popcnt_epi16
  // CHECK: @llvm.ctpop.v32i16
  return _mm512_popcnt_epi16(__A);
}
TEST_CONSTEXPR(match_v32hi(_mm512_popcnt_epi16((__m512i)(__v32hi){+5, -3, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129, +511, +1025, +5, -3, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129, +511, +1025}), 2, 15, 14, 1, 0, 8, 1, 9, 2, 2, 4, 2, 6, 2, 9, 2, 2, 15, 14, 1, 0, 8, 1, 9, 2, 2, 4, 2, 6, 2, 9, 2));

__m512i test_mm512_mask_popcnt_epi16(__m512i __A, __mmask32 __U, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_popcnt_epi16
  // CHECK: @llvm.ctpop.v32i16
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_popcnt_epi16(__A, __U, __B);
}
__m512i test_mm512_maskz_popcnt_epi16(__mmask32 __U, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_popcnt_epi16
  // CHECK: @llvm.ctpop.v32i16
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_popcnt_epi16(__U, __B);
}

__m512i test_mm512_popcnt_epi8(__m512i __A) {
  // CHECK-LABEL: test_mm512_popcnt_epi8
  // CHECK: @llvm.ctpop.v64i8
  return _mm512_popcnt_epi8(__A);
}
TEST_CONSTEXPR(match_v64qi(_mm512_popcnt_epi8((__m512i)(__v64qi){+5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73, +5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73, +5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73, +5, -3, -10, +8, 0, -16, +16, -16, +3, +9, +15, +33, +63, +33, +53, +73}), 2, 7, 6, 1, 0, 4, 1, 4, 2, 2, 4, 2, 6, 2, 4, 3, 2, 7, 6, 1, 0, 4, 1, 4, 2, 2, 4, 2, 6, 2, 4, 3, 2, 7, 6, 1, 0, 4, 1, 4, 2, 2, 4, 2, 6, 2, 4, 3, 2, 7, 6, 1, 0, 4, 1, 4, 2, 2, 4, 2, 6, 2, 4, 3));

__m512i test_mm512_mask_popcnt_epi8(__m512i __A, __mmask64 __U, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_popcnt_epi8
  // CHECK: @llvm.ctpop.v64i8
  // CHECK: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_popcnt_epi8(__A, __U, __B);
}
__m512i test_mm512_maskz_popcnt_epi8(__mmask64 __U, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_popcnt_epi8
  // CHECK: @llvm.ctpop.v64i8
  // CHECK: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_popcnt_epi8(__U, __B);
}

__mmask64 test_mm512_mask_bitshuffle_epi64_mask(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_bitshuffle_epi64_mask
  // CHECK: @llvm.x86.avx512.vpshufbitqmb.512
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_bitshuffle_epi64_mask(__U, __A, __B);
}

__mmask64 test_mm512_bitshuffle_epi64_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_bitshuffle_epi64_mask
  // CHECK: @llvm.x86.avx512.vpshufbitqmb.512
  return _mm512_bitshuffle_epi64_mask(__A, __B);
}

