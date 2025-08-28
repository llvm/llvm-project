// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vpopcntdq -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vpopcntdq -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vpopcntdq -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vpopcntdq -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m512i test_mm512_popcnt_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_popcnt_epi64
  // CHECK: @llvm.ctpop.v8i64
  return _mm512_popcnt_epi64(__A);
}
TEST_CONSTEXPR(match_v8di(_mm512_popcnt_epi64((__m512i)(__v8di){+5, -3, -10, +8, 0, -256, +256, -128}), 2, 63, 62, 1, 0, 56, 1, 57));

__m512i test_mm512_mask_popcnt_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_popcnt_epi64
  // CHECK: @llvm.ctpop.v8i64
  // CHECK: select <8 x i1> %{{[0-9]+}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_popcnt_epi64(__W, __U, __A);
}
TEST_CONSTEXPR(match_v8di(_mm512_mask_popcnt_epi64(_mm512_set1_epi64(-1), 0x81, (__m512i)(__v8di){+5, -3, -10, +8, 0, -256, +256, -128}), 2, -1, -1, -1, -1, -1, -1, 57));

__m512i test_mm512_maskz_popcnt_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_popcnt_epi64
  // CHECK: @llvm.ctpop.v8i64
  // CHECK: select <8 x i1> %{{[0-9]+}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_popcnt_epi64(__U, __A);
}
TEST_CONSTEXPR(match_v8di(_mm512_maskz_popcnt_epi64(0x42, (__m512i)(__v8di){+5, -3, -10, +8, 0, -256, +256, -128}), 0, 63, 0, 0, 0, 0, 1, 0));

__m512i test_mm512_popcnt_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_popcnt_epi32
  // CHECK: @llvm.ctpop.v16i32
  return _mm512_popcnt_epi32(__A);
}
TEST_CONSTEXPR(match_v16si(_mm512_popcnt_epi32((__m512i)(__v16si){+5, -3, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129, +511, +1025}), 2, 31, 30, 1, 0, 24, 1, 25, 2, 2, 4, 2, 6, 2, 9, 2));

__m512i test_mm512_mask_popcnt_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_popcnt_epi32
  // CHECK: @llvm.ctpop.v16i32
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_popcnt_epi32(__W, __U, __A);
}
TEST_CONSTEXPR(match_v16si(_mm512_mask_popcnt_epi32(_mm512_set1_epi32(-1), 0x0F81, (__m512i)(__v16si){+5, -3, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129, +511, +1025}), 2, -1, -1, -1, -1, -1, -1, 25, 2, 2, 4, 2, -1, -1, -1, -1));

__m512i test_mm512_maskz_popcnt_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_popcnt_epi32
  // CHECK: @llvm.ctpop.v16i32
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_popcnt_epi32(__U, __A);
}
TEST_CONSTEXPR(match_v16si(_mm512_maskz_popcnt_epi32(0xF042, (__m512i)(__v16si){+5, -3, -10, +8, 0, -256, +256, -128, +3, +9, +15, +33, +63, +129, +511, +1025}), 0, 31, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6, 2, 9, 2));
