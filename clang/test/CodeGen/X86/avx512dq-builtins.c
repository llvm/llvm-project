// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>
#include "builtin_test_helpers.h"

__mmask8 test_knot_mask8(__mmask8 a) {
  // CHECK-LABEL: test_knot_mask8
  // CHECK: [[IN:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[NOT:%.*]] = xor <8 x i1> [[IN]], splat (i1 true)
  return _knot_mask8(a);
}

__mmask8 test_kand_mask8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kand_mask8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = and <8 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu64_mask(_kand_mask8(_mm512_cmpneq_epu64_mask(__A, __B),
                                                   _mm512_cmpneq_epu64_mask(__C, __D)),
                                                   __E, __F);
}

__mmask8 test_kandn_mask8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kandn_mask8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[NOT:%.*]] = xor <8 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = and <8 x i1> [[NOT]], [[RHS]]
  return _mm512_mask_cmpneq_epu64_mask(_kandn_mask8(_mm512_cmpneq_epu64_mask(__A, __B),
                                                    _mm512_cmpneq_epu64_mask(__C, __D)),
                                                    __E, __F);
}

__mmask8 test_kor_mask8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kor_mask8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = or <8 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu64_mask(_kor_mask8(_mm512_cmpneq_epu64_mask(__A, __B),
                                                  _mm512_cmpneq_epu64_mask(__C, __D)),
                                                  __E, __F);
}

__mmask8 test_kxnor_mask8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kxnor_mask8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[NOT:%.*]] = xor <8 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = xor <8 x i1> [[NOT]], [[RHS]]
  return _mm512_mask_cmpneq_epu64_mask(_kxnor_mask8(_mm512_cmpneq_epu64_mask(__A, __B),
                                                    _mm512_cmpneq_epu64_mask(__C, __D)),
                                                    __E, __F);
}

__mmask8 test_kxor_mask8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kxor_mask8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = xor <8 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu64_mask(_kxor_mask8(_mm512_cmpneq_epu64_mask(__A, __B),
                                                   _mm512_cmpneq_epu64_mask(__C, __D)),
                                                   __E, __F);
}

unsigned char test_kortestz_mask8_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_kortestz_mask8_u8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[OR:%.*]] = or <8 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <8 x i1> [[OR]] to i8
  // CHECK: [[CMP:%.*]] = icmp eq i8 [[CAST]], 0
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  return _kortestz_mask8_u8(_mm512_cmpneq_epu64_mask(__A, __B),
                            _mm512_cmpneq_epu64_mask(__C, __D));
}

unsigned char test_kortestc_mask8_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_kortestc_mask8_u8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[OR:%.*]] = or <8 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <8 x i1> [[OR]] to i8
  // CHECK: [[CMP:%.*]] = icmp eq i8 [[CAST]], -1
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  return _kortestc_mask8_u8(_mm512_cmpneq_epu64_mask(__A, __B),
                            _mm512_cmpneq_epu64_mask(__C, __D));
}

unsigned char test_kortest_mask8_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, unsigned char *CF) {
  // CHECK-LABEL: test_kortest_mask8_u8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[OR:%.*]] = or <8 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <8 x i1> [[OR]] to i8
  // CHECK: [[CMP:%.*]] = icmp eq i8 [[CAST]], -1
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  // CHECK: [[LHS2:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS2:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[OR2:%.*]] = or <8 x i1> [[LHS2]], [[RHS2]]
  // CHECK: [[CAST2:%.*]] = bitcast <8 x i1> [[OR2]] to i8
  // CHECK: [[CMP2:%.*]] = icmp eq i8 [[CAST2]], 0
  // CHECK: [[ZEXT2:%.*]] = zext i1 [[CMP2]] to i32
  // CHECK: trunc i32 [[ZEXT2]] to i8
  return _kortest_mask8_u8(_mm512_cmpneq_epu64_mask(__A, __B),
                           _mm512_cmpneq_epu64_mask(__C, __D), CF);
}

unsigned char test_ktestz_mask8_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_ktestz_mask8_u8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestz.b(<8 x i1> [[LHS]], <8 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktestz_mask8_u8(_mm512_cmpneq_epu64_mask(__A, __B),
                          _mm512_cmpneq_epu64_mask(__C, __D));
}

unsigned char test_ktestc_mask8_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_ktestc_mask8_u8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestc.b(<8 x i1> [[LHS]], <8 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktestc_mask8_u8(_mm512_cmpneq_epu64_mask(__A, __B),
                          _mm512_cmpneq_epu64_mask(__C, __D));
}

unsigned char test_ktest_mask8_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, unsigned char *CF) {
  // CHECK-LABEL: test_ktest_mask8_u8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestc.b(<8 x i1> [[LHS]], <8 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestz.b(<8 x i1> [[LHS]], <8 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktest_mask8_u8(_mm512_cmpneq_epu64_mask(__A, __B),
                         _mm512_cmpneq_epu64_mask(__C, __D), CF);
}

unsigned char test_ktestz_mask16_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_ktestz_mask16_u8
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestz.w(<16 x i1> [[LHS]], <16 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktestz_mask16_u8(_mm512_cmpneq_epu32_mask(__A, __B),
                           _mm512_cmpneq_epu32_mask(__C, __D));
}

unsigned char test_ktestc_mask16_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: test_ktestc_mask16_u8
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestc.w(<16 x i1> [[LHS]], <16 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktestc_mask16_u8(_mm512_cmpneq_epu32_mask(__A, __B),
                           _mm512_cmpneq_epu32_mask(__C, __D));
}

unsigned char test_ktest_mask16_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, unsigned char *CF) {
  // CHECK-LABEL: test_ktest_mask16_u8
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestc.w(<16 x i1> [[LHS]], <16 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestz.w(<16 x i1> [[LHS]], <16 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktest_mask16_u8(_mm512_cmpneq_epu32_mask(__A, __B),
                          _mm512_cmpneq_epu32_mask(__C, __D), CF);
}

__mmask8 test_kadd_mask8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kadd_mask8
  // CHECK: [[LHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = call <8 x i1> @llvm.x86.avx512.kadd.b(<8 x i1> [[LHS]], <8 x i1> [[RHS]])
  return _mm512_mask_cmpneq_epu64_mask(_kadd_mask8(_mm512_cmpneq_epu64_mask(__A, __B),
                                                   _mm512_cmpneq_epu64_mask(__C, __D)),
                                                   __E, __F);
}

__mmask16 test_kadd_mask16(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: test_kadd_mask16
  // CHECK: [[LHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: [[RES:%.*]] = call <16 x i1> @llvm.x86.avx512.kadd.w(<16 x i1> [[LHS]], <16 x i1> [[RHS]])
  return _mm512_mask_cmpneq_epu32_mask(_kadd_mask16(_mm512_cmpneq_epu32_mask(__A, __B),
                                                    _mm512_cmpneq_epu32_mask(__C, __D)),
                                                    __E, __F);
}

__mmask8 test_kshiftli_mask8(__m512i A, __m512i B, __m512i C, __m512i D) {
  // CHECK-LABEL: test_kshiftli_mask8
  // CHECK: [[VAL:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = shufflevector <8 x i1> zeroinitializer, <8 x i1> [[VAL]], <8 x i32> <i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13>
  return _mm512_mask_cmpneq_epu64_mask(_kshiftli_mask8(_mm512_cmpneq_epu64_mask(A, B), 2), C, D);
}

__mmask8 test_kshiftri_mask8(__m512i A, __m512i B, __m512i C, __m512i D) {
  // CHECK-LABEL: test_kshiftri_mask8
  // CHECK: [[VAL:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: [[RES:%.*]] = shufflevector <8 x i1> [[VAL]], <8 x i1> zeroinitializer, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  return _mm512_mask_cmpneq_epu64_mask(_kshiftri_mask8(_mm512_cmpneq_epu64_mask(A, B), 2), C, D);
}

unsigned int test_cvtmask8_u32(__m512i A, __m512i B) {
  // CHECK-LABEL: test_cvtmask8_u32
  // CHECK: zext i8 %{{.*}} to i32
  return _cvtmask8_u32(_mm512_cmpneq_epu64_mask(A, B));
}

__mmask8 test_cvtu32_mask8(__m512i A, __m512i B, unsigned int C) {
  // CHECK-LABEL: test_cvtu32_mask8
  // CHECK: trunc i32 %{{.*}} to i8
  return _mm512_mask_cmpneq_epu64_mask(_cvtu32_mask8(C), A, B);
}

__mmask8 test_load_mask8(__mmask8 *A, __m512i B, __m512i C) {
  // CHECK-LABEL: test_load_mask8
  // CHECK: [[LOAD:%.*]] = load i8, ptr %{{.*}}
  return _mm512_mask_cmpneq_epu64_mask(_load_mask8(A), B, C);
}

void test_store_mask8(__mmask8 *A, __m512i B, __m512i C) {
  // CHECK-LABEL: test_store_mask8
  // CHECK: store i8 %{{.*}}, ptr %{{.*}}
  _store_mask8(A, _mm512_cmpneq_epu64_mask(B, C));
}

__m512i test_mm512_mullo_epi64 (__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mullo_epi64
  // CHECK: mul <8 x i64>
  return (__m512i) _mm512_mullo_epi64(__A, __B);
}

__m512i test_mm512_mask_mullo_epi64 (__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_mullo_epi64
  // CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return (__m512i) _mm512_mask_mullo_epi64(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_mullo_epi64 (__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_mullo_epi64
  // CHECK: mul <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return (__m512i) _mm512_maskz_mullo_epi64(__U, __A, __B);
}

__m512d test_mm512_xor_pd (__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_xor_pd
  // CHECK: xor <8 x i64>
  return (__m512d) _mm512_xor_pd(__A, __B);
}

__m512d test_mm512_mask_xor_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_xor_pd
  // CHECK: xor <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return (__m512d) _mm512_mask_xor_pd(__W, __U, __A, __B);
}
TEST_CONSTEXPR(match_m512d(_mm512_xor_pd((__m512d){-4.0, -5.0, +6.0, +7.0, +7.0, +6.0, -5.0, -4.0}, (__m512d){+0.0, -0.0, -0.0, +7.0, +7.0, -0.0, -0.0, +0.0}), -4.0, +5.0, -6.0, +0.0, +0.0, -6.0, +5.0, -4.0));

__m512d test_mm512_maskz_xor_pd (__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_xor_pd
  // CHECK: xor <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return (__m512d) _mm512_maskz_xor_pd(__U, __A, __B);
}

__m512 test_mm512_xor_ps (__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_xor_ps
  // CHECK: xor <16 x i32>
  return (__m512) _mm512_xor_ps(__A, __B);
}
TEST_CONSTEXPR(match_m512(_mm512_xor_ps((__m512){-4.0f, -5.0f, +6.0f, +7.0f, +7.0f, +6.0f, -5.0f, -4.0f, -4.0f, -5.0f, +6.0f, +7.0f, +7.0f, +6.0f, -5.0f, -4.0f}, (__m512){+0.0f, -0.0f, -0.0f, +7.0f, +7.0f, -0.0f, -0.0f, +0.0f, +0.0f, -0.0f, -0.0f, +7.0f, +7.0f, -0.0f, -0.0f, +0.0f}), -4.0f, +5.0f, -6.0f, +0.0f, +0.0f, -6.0f, +5.0f, -4.0f, -4.0f, +5.0f, -6.0f, +0.0f, +0.0f, -6.0f, +5.0f, -4.0f));

__m512 test_mm512_mask_xor_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_xor_ps
  // CHECK: xor <16 x i32>
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return (__m512) _mm512_mask_xor_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_xor_ps (__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_xor_ps
  // CHECK: xor <16 x i32>
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return (__m512) _mm512_maskz_xor_ps(__U, __A, __B);
}

__m512d test_mm512_or_pd (__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_or_pd
  // CHECK: or <8 x i64>
  return (__m512d) _mm512_or_pd(__A, __B);
}
TEST_CONSTEXPR(match_m512d(_mm512_or_pd((__m512d){-4.0, -5.0, +6.0, +7.0, +7.0, +6.0, -5.0, -4.0}, (__m512d){+0.0, -0.0, -0.0, +7.0, +7.0, -0.0, -0.0, +0.0}), -4.0, -5.0, -6.0, +7.0, +7.0, -6.0, -5.0, -4.0));

__m512d test_mm512_mask_or_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_or_pd
  // CHECK: or <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return (__m512d) _mm512_mask_or_pd(__W, __U, __A, __B);
}

__m512d test_mm512_maskz_or_pd (__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_or_pd
  // CHECK: or <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return (__m512d) _mm512_maskz_or_pd(__U, __A, __B);
}

__m512 test_mm512_or_ps (__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_or_ps
  // CHECK: or <16 x i32>
  return (__m512) _mm512_or_ps(__A, __B);
}
TEST_CONSTEXPR(match_m512(_mm512_or_ps((__m512){-4.0f, -5.0f, +6.0f, +7.0f, +7.0f, +6.0f, -5.0f, -4.0f, -4.0f, -5.0f, +6.0f, +7.0f, +7.0f, +6.0f, -5.0f, -4.0f}, (__m512){+0.0f, -0.0f, -0.0f, +7.0f, +7.0f, -0.0f, -0.0f, +0.0f, +0.0f, -0.0f, -0.0f, +7.0f, +7.0f, -0.0f, -0.0f, +0.0f}), -4.0f, -5.0f, -6.0f, +7.0f, +7.0f, -6.0f, -5.0f, -4.0f, -4.0f, -5.0f, -6.0f, +7.0f, +7.0f, -6.0f, -5.0f, -4.0f));

__m512 test_mm512_mask_or_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_or_ps
  // CHECK: or <16 x i32>
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return (__m512) _mm512_mask_or_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_or_ps (__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_or_ps
  // CHECK: or <16 x i32>
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return (__m512) _mm512_maskz_or_ps(__U, __A, __B);
}

__m512d test_mm512_and_pd (__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_and_pd
  // CHECK: and <8 x i64>
  return (__m512d) _mm512_and_pd(__A, __B);
}
TEST_CONSTEXPR(match_m512d(_mm512_and_pd((__m512d){-4.0, -5.0, +6.0, +7.0, +7.0, +6.0, -5.0, -4.0}, (__m512d){+0.0, -0.0, -0.0, +7.0, +7.0, -0.0, -0.0, +0.0}), +0.0, -0.0, +0.0, +7.0, +7.0, +0.0, -0.0, +0.0));

__m512d test_mm512_mask_and_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_and_pd
  // CHECK: and <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return (__m512d) _mm512_mask_and_pd(__W, __U, __A, __B);
}

__m512d test_mm512_maskz_and_pd (__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_and_pd
  // CHECK: and <8 x i64>
  // CHECK: %[[MASK:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: select <8 x i1> %[[MASK]], <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return (__m512d) _mm512_maskz_and_pd(__U, __A, __B);
}

__m512 test_mm512_and_ps (__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_and_ps
  // CHECK: and <16 x i32>
  return (__m512) _mm512_and_ps(__A, __B);
}
TEST_CONSTEXPR(match_m512(_mm512_and_ps((__m512){-4.0f, -5.0f, +6.0f, +7.0f, +7.0f, +6.0f, -5.0f, -4.0f, -4.0f, -5.0f, +6.0f, +7.0f, +7.0f, +6.0f, -5.0f, -4.0f}, (__m512){+0.0f, -0.0f, -0.0f, +7.0f, +7.0f, -0.0f, -0.0f, +0.0f, +0.0f, -0.0f, -0.0f, +7.0f, +7.0f, -0.0f, -0.0f, +0.0f}), +0.0f, -0.0f, +0.0f, +7.0f, +7.0f, +0.0f, -0.0f, +0.0f, +0.0f, -0.0f, +0.0f, +7.0f, +7.0f, +0.0f, -0.0f, +0.0f));

__m512 test_mm512_mask_and_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_and_ps
  // CHECK: and <16 x i32>
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return (__m512) _mm512_mask_and_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_and_ps (__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_and_ps
  // CHECK: and <16 x i32>
  // CHECK: %[[MASK:.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: select <16 x i1> %[[MASK]], <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return (__m512) _mm512_maskz_and_ps(__U, __A, __B);
}

__m512d test_mm512_andnot_pd (__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_andnot_pd
  // CHECK: xor <8 x i64> %{{.*}}, splat (i64 -1)
  // CHECK: and <8 x i64>
  return (__m512d) _mm512_andnot_pd(__A, __B);
}
TEST_CONSTEXPR(match_m512d(_mm512_andnot_pd((__m512d){-4.0, -5.0, +6.0, +7.0, +7.0, +6.0, -5.0, -4.0}, (__m512d){+0.0, -0.0, -0.0, +7.0, +7.0, -0.0, -0.0, +0.0}), +0.0, +0.0, -0.0, +0.0, +0.0, -0.0, +0.0, +0.0));

__m512d test_mm512_mask_andnot_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_andnot_pd
  // CHECK: xor <8 x i64> %{{.*}}, splat (i64 -1)
  // CHECK: and <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return (__m512d) _mm512_mask_andnot_pd(__W, __U, __A, __B);
}

__m512d test_mm512_maskz_andnot_pd (__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_andnot_pd
  // CHECK: xor <8 x i64> %{{.*}}, splat (i64 -1)
  // CHECK: and <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return (__m512d) _mm512_maskz_andnot_pd(__U, __A, __B);
}

__m512 test_mm512_andnot_ps (__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_andnot_ps
  // CHECK: xor <16 x i32> %{{.*}}, splat (i32 -1)
  // CHECK: and <16 x i32>
  return (__m512) _mm512_andnot_ps(__A, __B);
}
TEST_CONSTEXPR(match_m512(_mm512_andnot_ps((__m512){-4.0f, -5.0f, +6.0f, +7.0f, +7.0f, +6.0f, -5.0f, -4.0f, -4.0f, -5.0f, +6.0f, +7.0f, +7.0f, +6.0f, -5.0f, -4.0f}, (__m512){+0.0f, -0.0f, -0.0f, +7.0f, +7.0f, -0.0f, -0.0f, +0.0f, +0.0f, -0.0f, -0.0f, +7.0f, +7.0f, -0.0f, -0.0f, +0.0f}), +0.0f, +0.0f, -0.0f, +0.0f, +0.0f, -0.0f, +0.0f, +0.0f, +0.0f, +0.0f, -0.0f, +0.0f, +0.0f, -0.0f, +0.0f, +0.0f));

__m512 test_mm512_mask_andnot_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_andnot_ps
  // CHECK: xor <16 x i32> %{{.*}}, splat (i32 -1)
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return (__m512) _mm512_mask_andnot_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_andnot_ps (__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_andnot_ps
  // CHECK: xor <16 x i32> %{{.*}}, splat (i32 -1)
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return (__m512) _mm512_maskz_andnot_ps(__U, __A, __B);
}

__m512i test_mm512_cvtpd_epi64(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.512
  return _mm512_cvtpd_epi64(__A); 
}

__m512i test_mm512_mask_cvtpd_epi64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.512
  return _mm512_mask_cvtpd_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtpd_epi64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.512
  return _mm512_maskz_cvtpd_epi64(__U, __A); 
}

__m512i test_mm512_cvt_roundpd_epi64(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.512
  return _mm512_cvt_roundpd_epi64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_mask_cvt_roundpd_epi64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.512
  return _mm512_mask_cvt_roundpd_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_maskz_cvt_roundpd_epi64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.512
  return _mm512_maskz_cvt_roundpd_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_cvtpd_epu64(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.512
  return _mm512_cvtpd_epu64(__A); 
}

__m512i test_mm512_mask_cvtpd_epu64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.512
  return _mm512_mask_cvtpd_epu64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtpd_epu64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.512
  return _mm512_maskz_cvtpd_epu64(__U, __A); 
}

__m512i test_mm512_cvt_roundpd_epu64(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.512
  return _mm512_cvt_roundpd_epu64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_mask_cvt_roundpd_epu64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.512
  return _mm512_mask_cvt_roundpd_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_maskz_cvt_roundpd_epu64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.512
  return _mm512_maskz_cvt_roundpd_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_cvtps_epi64(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.512
  return _mm512_cvtps_epi64(__A); 
}

__m512i test_mm512_mask_cvtps_epi64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.512
  return _mm512_mask_cvtps_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtps_epi64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.512
  return _mm512_maskz_cvtps_epi64(__U, __A); 
}

__m512i test_mm512_cvt_roundps_epi64(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.512
  return _mm512_cvt_roundps_epi64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_mask_cvt_roundps_epi64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.512
  return _mm512_mask_cvt_roundps_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_maskz_cvt_roundps_epi64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.512
  return _mm512_maskz_cvt_roundps_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_cvtps_epu64(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.512
  return _mm512_cvtps_epu64(__A); 
}

__m512i test_mm512_mask_cvtps_epu64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.512
  return _mm512_mask_cvtps_epu64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtps_epu64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.512
  return _mm512_maskz_cvtps_epu64(__U, __A); 
}

__m512i test_mm512_cvt_roundps_epu64(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.512
  return _mm512_cvt_roundps_epu64(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_mask_cvt_roundps_epu64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.512
  return _mm512_mask_cvt_roundps_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_maskz_cvt_roundps_epu64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.512
  return _mm512_maskz_cvt_roundps_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512d test_mm512_cvtepi64_pd(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepi64_pd
  // CHECK: sitofp <8 x i64> %{{.*}} to <8 x double>
  return _mm512_cvtepi64_pd(__A); 
}

__m512d test_mm512_mask_cvtepi64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_pd
  // CHECK: sitofp <8 x i64> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_cvtepi64_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_cvtepi64_pd(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi64_pd
  // CHECK: sitofp <8 x i64> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_cvtepi64_pd(__U, __A); 
}

__m512d test_mm512_cvt_roundepi64_pd(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvt_roundepi64_pd
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f64.v8i64
  return _mm512_cvt_roundepi64_pd(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512d test_mm512_mask_cvt_roundepi64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepi64_pd
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f64.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_cvt_roundepi64_pd(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512d test_mm512_maskz_cvt_roundepi64_pd(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepi64_pd
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f64.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_cvt_roundepi64_pd(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m256 test_mm512_cvtepi64_ps(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepi64_ps
  // CHECK: sitofp <8 x i64> %{{.*}} to <8 x float>
  return _mm512_cvtepi64_ps(__A); 
}

__m256 test_mm512_mask_cvtepi64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_ps
  // CHECK: sitofp <8 x i64> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_mask_cvtepi64_ps(__W, __U, __A); 
}

__m256 test_mm512_maskz_cvtepi64_ps(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi64_ps
  // CHECK: sitofp <8 x i64> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_maskz_cvtepi64_ps(__U, __A); 
}

__m256 test_mm512_cvt_roundepi64_ps(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvt_roundepi64_ps
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f32.v8i64
  return _mm512_cvt_roundepi64_ps(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m256 test_mm512_mask_cvt_roundepi64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepi64_ps
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f32.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_mask_cvt_roundepi64_ps(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m256 test_mm512_maskz_cvt_roundepi64_ps(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepi64_ps
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f32.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_maskz_cvt_roundepi64_ps(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512i test_mm512_cvttpd_epi64(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.512
  return _mm512_cvttpd_epi64(__A); 
}

__m512i test_mm512_mask_cvttpd_epi64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.512
  return _mm512_mask_cvttpd_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvttpd_epi64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.512
  return _mm512_maskz_cvttpd_epi64(__U, __A); 
}

__m512i test_mm512_cvtt_roundpd_epi64(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvtt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.512
  return _mm512_cvtt_roundpd_epi64(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundpd_epi64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.512
  return _mm512_mask_cvtt_roundpd_epi64(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundpd_epi64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.512
  return _mm512_maskz_cvtt_roundpd_epi64(__U, __A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttpd_epu64(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.512
  return _mm512_cvttpd_epu64(__A); 
}

__m512i test_mm512_mask_cvttpd_epu64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.512
  return _mm512_mask_cvttpd_epu64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvttpd_epu64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.512
  return _mm512_maskz_cvttpd_epu64(__U, __A); 
}

__m512i test_mm512_cvtt_roundpd_epu64(__m512d __A) {
  // CHECK-LABEL: test_mm512_cvtt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.512
  return _mm512_cvtt_roundpd_epu64(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundpd_epu64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.512
  return _mm512_mask_cvtt_roundpd_epu64(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundpd_epu64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.512
  return _mm512_maskz_cvtt_roundpd_epu64(__U, __A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttps_epi64(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.512
  return _mm512_cvttps_epi64(__A); 
}

__m512i test_mm512_mask_cvttps_epi64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.512
  return _mm512_mask_cvttps_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvttps_epi64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.512
  return _mm512_maskz_cvttps_epi64(__U, __A); 
}

__m512i test_mm512_cvtt_roundps_epi64(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvtt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.512
  return _mm512_cvtt_roundps_epi64(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundps_epi64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.512
  return _mm512_mask_cvtt_roundps_epi64(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundps_epi64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.512
  return _mm512_maskz_cvtt_roundps_epi64(__U, __A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttps_epu64(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.512
  return _mm512_cvttps_epu64(__A); 
}

__m512i test_mm512_mask_cvttps_epu64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.512
  return _mm512_mask_cvttps_epu64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvttps_epu64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.512
  return _mm512_maskz_cvttps_epu64(__U, __A); 
}

__m512i test_mm512_cvtt_roundps_epu64(__m256 __A) {
  // CHECK-LABEL: test_mm512_cvtt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.512
  return _mm512_cvtt_roundps_epu64(__A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundps_epu64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.512
  return _mm512_mask_cvtt_roundps_epu64(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundps_epu64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.512
  return _mm512_maskz_cvtt_roundps_epu64(__U, __A, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_cvtepu64_pd(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepu64_pd
  // CHECK: uitofp <8 x i64> %{{.*}} to <8 x double>
  return _mm512_cvtepu64_pd(__A); 
}

__m512d test_mm512_mask_cvtepu64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepu64_pd
  // CHECK: uitofp <8 x i64> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_cvtepu64_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_cvtepu64_pd(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu64_pd
  // CHECK: uitofp <8 x i64> %{{.*}} to <8 x double>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_cvtepu64_pd(__U, __A); 
}

__m512d test_mm512_cvt_roundepu64_pd(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvt_roundepu64_pd
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f64.v8i64
  return _mm512_cvt_roundepu64_pd(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512d test_mm512_mask_cvt_roundepu64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepu64_pd
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f64.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_cvt_roundepu64_pd(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512d test_mm512_maskz_cvt_roundepu64_pd(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepu64_pd
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f64.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_cvt_roundepu64_pd(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m256 test_mm512_cvtepu64_ps(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvtepu64_ps
  // CHECK: uitofp <8 x i64> %{{.*}} to <8 x float>
  return _mm512_cvtepu64_ps(__A); 
}

__m256 test_mm512_mask_cvtepu64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvtepu64_ps
  // CHECK: uitofp <8 x i64> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_mask_cvtepu64_ps(__W, __U, __A); 
}

__m256 test_mm512_maskz_cvtepu64_ps(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu64_ps
  // CHECK: uitofp <8 x i64> %{{.*}} to <8 x float>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_maskz_cvtepu64_ps(__U, __A); 
}

__m256 test_mm512_cvt_roundepu64_ps(__m512i __A) {
  // CHECK-LABEL: test_mm512_cvt_roundepu64_ps
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f32.v8i64
  return _mm512_cvt_roundepu64_ps(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m256 test_mm512_mask_cvt_roundepu64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepu64_ps
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f32.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_mask_cvt_roundepu64_ps(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m256 test_mm512_maskz_cvt_roundepu64_ps(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepu64_ps
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f32.v8i64
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_maskz_cvt_roundepu64_ps(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
}

__m512d test_mm512_range_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_range_pd(__A, __B, 4); 
}

__m512d test_mm512_mask_range_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_mask_range_pd(__W, __U, __A, __B, 4); 
}

__m512d test_mm512_maskz_range_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_maskz_range_pd(__U, __A, __B, 4); 
}

__m512d test_mm512_range_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_range_round_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_range_round_pd(__A, __B, 4, 8); 
}

__m512d test_mm512_mask_range_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_mask_range_round_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_mask_range_round_pd(__W, __U, __A, __B, 4, 8); 
}

__m512d test_mm512_maskz_range_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: test_mm512_maskz_range_round_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_maskz_range_round_pd(__U, __A, __B, 4, 8); 
}

__m128d test_mm512_range_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm512_range_round_sd
  // CHECK: @llvm.x86.avx512.mask.range.sd
  return _mm_range_round_sd(__A, __B, 4, 8); 
}

__m128d test_mm512_mask_range_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm512_mask_range_round_sd
  // CHECK: @llvm.x86.avx512.mask.range.sd
  return _mm_mask_range_round_sd(__W, __U, __A, __B, 4, 8); 
}

__m128d test_mm512_maskz_range_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm512_maskz_range_round_sd
  // CHECK: @llvm.x86.avx512.mask.range.sd
  return _mm_maskz_range_round_sd(__U, __A, __B, 4, 8); 
}

__m128 test_mm512_range_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm512_range_round_ss
  // CHECK: @llvm.x86.avx512.mask.range.ss
  return _mm_range_round_ss(__A, __B, 4, 8); 
}

__m128 test_mm512_mask_range_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm512_mask_range_round_ss
  // CHECK: @llvm.x86.avx512.mask.range.ss
  return _mm_mask_range_round_ss(__W, __U, __A, __B, 4, 8); 
}

__m128 test_mm512_maskz_range_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm512_maskz_range_round_ss
  // CHECK: @llvm.x86.avx512.mask.range.ss
  return _mm_maskz_range_round_ss(__U, __A, __B, 4, 8); 
}

__m128d test_mm_range_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_range_sd
  // CHECK: @llvm.x86.avx512.mask.range.sd
  return _mm_range_sd(__A, __B, 4); 
}

__m128d test_mm_mask_range_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_range_sd
  // CHECK: @llvm.x86.avx512.mask.range.sd
  return _mm_mask_range_sd(__W, __U, __A, __B, 4); 
}

__m128d test_mm_maskz_range_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_range_sd
  // CHECK: @llvm.x86.avx512.mask.range.sd
  return _mm_maskz_range_sd(__U, __A, __B, 4); 
}

__m128 test_mm_range_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_range_ss
  // CHECK: @llvm.x86.avx512.mask.range.ss
  return _mm_range_ss(__A, __B, 4); 
}

__m128 test_mm_mask_range_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_range_ss
  // CHECK: @llvm.x86.avx512.mask.range.ss
  return _mm_mask_range_ss(__W, __U, __A, __B, 4); 
}

__m128 test_mm_maskz_range_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_range_ss
  // CHECK: @llvm.x86.avx512.mask.range.ss
  return _mm_maskz_range_ss(__U, __A, __B, 4); 
}

__m512 test_mm512_range_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_range_ps(__A, __B, 4); 
}

__m512 test_mm512_mask_range_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_mask_range_ps(__W, __U, __A, __B, 4); 
}

__m512 test_mm512_maskz_range_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_maskz_range_ps(__U, __A, __B, 4); 
}

__m512 test_mm512_range_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_range_round_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_range_round_ps(__A, __B, 4, 8); 
}

__m512 test_mm512_mask_range_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_mask_range_round_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_mask_range_round_ps(__W, __U, __A, __B, 4, 8); 
}

__m512 test_mm512_maskz_range_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: test_mm512_maskz_range_round_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_maskz_range_round_ps(__U, __A, __B, 4, 8); 
}

__m512d test_mm512_reduce_pd(__m512d __A) {
  // CHECK-LABEL: test_mm512_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_reduce_pd(__A, 4); 
}

__m512d test_mm512_mask_reduce_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_mask_reduce_pd(__W, __U, __A, 4); 
}

__m512d test_mm512_maskz_reduce_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_maskz_reduce_pd(__U, __A, 4); 
}

__m512 test_mm512_reduce_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_reduce_ps(__A, 4); 
}

__m512 test_mm512_mask_reduce_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_mask_reduce_ps(__W, __U, __A, 4); 
}

__m512 test_mm512_maskz_reduce_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_maskz_reduce_ps(__U, __A, 4); 
}

__m512d test_mm512_reduce_round_pd(__m512d __A) {
  // CHECK-LABEL: test_mm512_reduce_round_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_reduce_round_pd(__A, 4, 8); 
}

__m512d test_mm512_mask_reduce_round_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_reduce_round_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_mask_reduce_round_pd(__W, __U, __A, 4, 8); 
}

__m512d test_mm512_maskz_reduce_round_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_reduce_round_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_maskz_reduce_round_pd(__U, __A, 4, 8);
}

__m512 test_mm512_reduce_round_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_reduce_round_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_reduce_round_ps(__A, 4, 8); 
}

__m512 test_mm512_mask_reduce_round_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_reduce_round_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_mask_reduce_round_ps(__W, __U, __A, 4, 8); 
}

__m512 test_mm512_maskz_reduce_round_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_reduce_round_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_maskz_reduce_round_ps(__U, __A, 4, 8); 
}

__m128 test_mm_reduce_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_reduce_ss
  // CHECK: @llvm.x86.avx512.mask.reduce.ss
  return _mm_reduce_ss(__A, __B, 4);
}

__m128 test_mm_mask_reduce_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_reduce_ss
  // CHECK: @llvm.x86.avx512.mask.reduce.ss
  return _mm_mask_reduce_ss(__W, __U, __A, __B, 4);
}

__m128 test_mm_maskz_reduce_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_reduce_ss
  // CHECK: @llvm.x86.avx512.mask.reduce.ss
  return _mm_maskz_reduce_ss(__U, __A, __B, 4);
}

__m128 test_mm_reduce_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_reduce_round_ss
  // CHECK: @llvm.x86.avx512.mask.reduce.ss
  return _mm_reduce_round_ss(__A, __B, 4, 8);
}

__m128 test_mm_mask_reduce_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_mask_reduce_round_ss
  // CHECK: @llvm.x86.avx512.mask.reduce.ss
  return _mm_mask_reduce_round_ss(__W, __U, __A, __B, 4, 8);
}

__m128 test_mm_maskz_reduce_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: test_mm_maskz_reduce_round_ss
  // CHECK: @llvm.x86.avx512.mask.reduce.ss
  return _mm_maskz_reduce_round_ss(__U, __A, __B, 4, 8);
}

__m128d test_mm_reduce_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_reduce_sd
  // CHECK: @llvm.x86.avx512.mask.reduce.sd
  return _mm_reduce_sd(__A, __B, 4);
}

__m128d test_mm_mask_reduce_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_reduce_sd
  // CHECK: @llvm.x86.avx512.mask.reduce.sd
  return _mm_mask_reduce_sd(__W, __U, __A, __B, 4);
}

__m128d test_mm_maskz_reduce_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_reduce_sd
  // CHECK: @llvm.x86.avx512.mask.reduce.sd
  return _mm_maskz_reduce_sd(__U, __A, __B, 4);
}

__m128d test_mm_reduce_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_reduce_round_sd
  // CHECK: @llvm.x86.avx512.mask.reduce.sd
  return _mm_reduce_round_sd(__A, __B, 4, 8);
}

__m128d test_mm_mask_reduce_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_mask_reduce_round_sd
  // CHECK: @llvm.x86.avx512.mask.reduce.sd
  return _mm_mask_reduce_round_sd(__W, __U, __A, __B, 4, 8);
}

__m128d test_mm_maskz_reduce_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: test_mm_maskz_reduce_round_sd
  // CHECK: @llvm.x86.avx512.mask.reduce.sd
  return _mm_maskz_reduce_round_sd(__U, __A, __B, 4, 8);
}

__mmask16 test_mm512_movepi32_mask(__m512i __A) {
  // CHECK-LABEL: test_mm512_movepi32_mask
  // CHECK: [[CMP:%.*]] = icmp slt <16 x i32> %{{.*}}, zeroinitializer
  return _mm512_movepi32_mask(__A); 
}

__m512i test_mm512_movm_epi32(__mmask16 __A) {
  // CHECK-LABEL: test_mm512_movm_epi32
  // CHECK: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: %vpmovm2.i = sext <16 x i1> %{{.*}} to <16 x i32>
  return _mm512_movm_epi32(__A); 
}

__m512i test_mm512_movm_epi64(__mmask8 __A) {
  // CHECK-LABEL: test_mm512_movm_epi64
  // CHECK: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: %vpmovm2.i = sext <8 x i1> %{{.*}} to <8 x i64>
  return _mm512_movm_epi64(__A); 
}

__mmask8 test_mm512_movepi64_mask(__m512i __A) {
  // CHECK-LABEL: test_mm512_movepi64_mask
  // CHECK: [[CMP:%.*]] = icmp slt <8 x i64> %{{.*}}, zeroinitializer
  return _mm512_movepi64_mask(__A); 
}

__m512 test_mm512_broadcast_f32x2(__m128 __A) {
  // CHECK-LABEL: test_mm512_broadcast_f32x2
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  return _mm512_broadcast_f32x2(__A); 
}

__m512 test_mm512_mask_broadcast_f32x2(__m512 __O, __mmask16 __M, __m128 __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_f32x2
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_broadcast_f32x2(__O, __M, __A); 
}

__m512 test_mm512_maskz_broadcast_f32x2(__mmask16 __M, __m128 __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_f32x2
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_broadcast_f32x2(__M, __A); 
}

__m512 test_mm512_broadcast_f32x8(float const* __A) {
  // CHECK-LABEL: test_mm512_broadcast_f32x8
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_broadcast_f32x8(_mm256_loadu_ps(__A)); 
}

__m512 test_mm512_mask_broadcast_f32x8(__m512 __O, __mmask16 __M, float const* __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_f32x8
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_broadcast_f32x8(__O, __M, _mm256_loadu_ps(__A)); 
}

__m512 test_mm512_maskz_broadcast_f32x8(__mmask16 __M, float const* __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_f32x8
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_broadcast_f32x8(__M, _mm256_loadu_ps(__A)); 
}

__m512d test_mm512_broadcast_f64x2(double const* __A) {
  // CHECK-LABEL: test_mm512_broadcast_f64x2
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  return _mm512_broadcast_f64x2(_mm_loadu_pd(__A)); 
}

__m512d test_mm512_mask_broadcast_f64x2(__m512d __O, __mmask8 __M, double const* __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_f64x2
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_broadcast_f64x2(__O, __M, _mm_loadu_pd(__A)); 
}

__m512d test_mm512_maskz_broadcast_f64x2(__mmask8 __M, double const* __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_f64x2
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_broadcast_f64x2(__M, _mm_loadu_pd(__A)); 
}

__m512i test_mm512_broadcast_i32x2(__m128i __A) {
  // CHECK-LABEL: test_mm512_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  return _mm512_broadcast_i32x2(__A); 
}

__m512i test_mm512_mask_broadcast_i32x2(__m512i __O, __mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_broadcast_i32x2(__O, __M, __A); 
}

__m512i test_mm512_maskz_broadcast_i32x2(__mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_broadcast_i32x2(__M, __A); 
}

__m512i test_mm512_broadcast_i32x8(__m256i const* __A) {
  // CHECK-LABEL: test_mm512_broadcast_i32x8
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_broadcast_i32x8(_mm256_loadu_si256(__A)); 
}

__m512i test_mm512_mask_broadcast_i32x8(__m512i __O, __mmask16 __M, __m256i const* __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_i32x8
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_broadcast_i32x8(__O, __M, _mm256_loadu_si256(__A)); 
}

__m512i test_mm512_maskz_broadcast_i32x8(__mmask16 __M, __m256i const* __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_i32x8
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_broadcast_i32x8(__M, _mm256_loadu_si256(__A)); 
}

__m512i test_mm512_broadcast_i64x2(__m128i const* __A) {
  // CHECK-LABEL: test_mm512_broadcast_i64x2
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  return _mm512_broadcast_i64x2(_mm_loadu_si128(__A)); 
}

__m512i test_mm512_mask_broadcast_i64x2(__m512i __O, __mmask8 __M, __m128i const* __A) {
  // CHECK-LABEL: test_mm512_mask_broadcast_i64x2
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_broadcast_i64x2(__O, __M, _mm_loadu_si128(__A)); 
}

__m512i test_mm512_maskz_broadcast_i64x2(__mmask8 __M, __m128i const* __A) {
  // CHECK-LABEL: test_mm512_maskz_broadcast_i64x2
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_broadcast_i64x2(__M, _mm_loadu_si128(__A)); 
}

__m256 test_mm512_extractf32x8_ps(__m512 __A) {
  // CHECK-LABEL: test_mm512_extractf32x8_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_extractf32x8_ps(__A, 1); 
}

__m256 test_mm512_mask_extractf32x8_ps(__m256 __W, __mmask8 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_extractf32x8_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_mask_extractf32x8_ps(__W, __U, __A, 1); 
}

__m256 test_mm512_maskz_extractf32x8_ps(__mmask8 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_maskz_extractf32x8_ps
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm512_maskz_extractf32x8_ps(__U, __A, 1); 
}

__m128d test_mm512_extractf64x2_pd(__m512d __A) {
  // CHECK-LABEL: test_mm512_extractf64x2_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <2 x i32> <i32 6, i32 7>
  return _mm512_extractf64x2_pd(__A, 3); 
}

__m128d test_mm512_mask_extractf64x2_pd(__m128d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_extractf64x2_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <2 x i32> <i32 6, i32 7>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm512_mask_extractf64x2_pd(__W, __U, __A, 3); 
}

__m128d test_mm512_maskz_extractf64x2_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_maskz_extractf64x2_pd
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> poison, <2 x i32> <i32 6, i32 7>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm512_maskz_extractf64x2_pd(__U, __A, 3); 
}

__m256i test_mm512_extracti32x8_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_extracti32x8_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_extracti32x8_epi32(__A, 1); 
}

__m256i test_mm512_mask_extracti32x8_epi32(__m256i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_extracti32x8_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm512_mask_extracti32x8_epi32(__W, __U, __A, 1); 
}

__m256i test_mm512_maskz_extracti32x8_epi32(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_extracti32x8_epi32
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm512_maskz_extracti32x8_epi32(__U, __A, 1); 
}

__m128i test_mm512_extracti64x2_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_extracti64x2_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <2 x i32> <i32 6, i32 7>
  return _mm512_extracti64x2_epi64(__A, 3); 
}

__m128i test_mm512_mask_extracti64x2_epi64(__m128i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_extracti64x2_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <2 x i32> <i32 6, i32 7>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm512_mask_extracti64x2_epi64(__W, __U, __A, 3); 
}

__m128i test_mm512_maskz_extracti64x2_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_extracti64x2_epi64
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> poison, <2 x i32> <i32 6, i32 7>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm512_maskz_extracti64x2_epi64(__U, __A, 3); 
}

__m512 test_mm512_insertf32x8(__m512 __A, __m256 __B) {
  // CHECK-LABEL: test_mm512_insertf32x8
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  return _mm512_insertf32x8(__A, __B, 1); 
}

__m512 test_mm512_mask_insertf32x8(__m512 __W, __mmask16 __U, __m512 __A, __m256 __B) {
  // CHECK-LABEL: test_mm512_mask_insertf32x8
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_insertf32x8(__W, __U, __A, __B, 1); 
}

__m512 test_mm512_maskz_insertf32x8(__mmask16 __U, __m512 __A, __m256 __B) {
  // CHECK-LABEL: test_mm512_maskz_insertf32x8
  // CHECK: shufflevector <16 x float> %{{.*}}, <16 x float> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_maskz_insertf32x8(__U, __A, __B, 1); 
}

__m512d test_mm512_insertf64x2(__m512d __A, __m128d __B) {
  // CHECK-LABEL: test_mm512_insertf64x2
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  return _mm512_insertf64x2(__A, __B, 3); 
}

__m512d test_mm512_mask_insertf64x2(__m512d __W, __mmask8 __U, __m512d __A, __m128d __B) {
  // CHECK-LABEL: test_mm512_mask_insertf64x2
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_mask_insertf64x2(__W, __U, __A, __B, 3); 
}

__m512d test_mm512_maskz_insertf64x2(__mmask8 __U, __m512d __A, __m128d __B) {
  // CHECK-LABEL: test_mm512_maskz_insertf64x2
  // CHECK: shufflevector <8 x double> %{{.*}}, <8 x double> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  // CHECK: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  return _mm512_maskz_insertf64x2(__U, __A, __B, 3); 
}

__m512i test_mm512_inserti32x8(__m512i __A, __m256i __B) {
  // CHECK-LABEL: test_mm512_inserti32x8
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  return _mm512_inserti32x8(__A, __B, 1); 
}

__m512i test_mm512_mask_inserti32x8(__m512i __W, __mmask16 __U, __m512i __A, __m256i __B) {
  // CHECK-LABEL: test_mm512_mask_inserti32x8
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_inserti32x8(__W, __U, __A, __B, 1); 
}

__m512i test_mm512_maskz_inserti32x8(__mmask16 __U, __m512i __A, __m256i __B) {
  // CHECK-LABEL: test_mm512_maskz_inserti32x8
  // CHECK: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_inserti32x8(__U, __A, __B, 1); 
}

__m512i test_mm512_inserti64x2(__m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_inserti64x2
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 4, i32 5, i32 6, i32 7>
  return _mm512_inserti64x2(__A, __B, 1); 
}

__m512i test_mm512_mask_inserti64x2(__m512i __W, __mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_mask_inserti64x2
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_inserti64x2(__W, __U, __A, __B, 1); 
}

__m512i test_mm512_maskz_inserti64x2(__mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: test_mm512_maskz_inserti64x2
  // CHECK: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_inserti64x2(__U, __A, __B, 1); 
}
__mmask8 test_mm512_mask_fpclass_pd_mask(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: test_mm512_mask_fpclass_pd_mask
  // CHECK: @llvm.x86.avx512.fpclass.pd.512
  return _mm512_mask_fpclass_pd_mask(__U, __A, 4); 
}

__mmask8 test_mm512_fpclass_pd_mask(__m512d __A) {
  // CHECK-LABEL: test_mm512_fpclass_pd_mask
  // CHECK: @llvm.x86.avx512.fpclass.pd.512
  return _mm512_fpclass_pd_mask(__A, 4); 
}

__mmask16 test_mm512_mask_fpclass_ps_mask(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: test_mm512_mask_fpclass_ps_mask
  // CHECK: @llvm.x86.avx512.fpclass.ps.512
  return _mm512_mask_fpclass_ps_mask(__U, __A, 4); 
}

__mmask16 test_mm512_fpclass_ps_mask(__m512 __A) {
  // CHECK-LABEL: test_mm512_fpclass_ps_mask
  // CHECK: @llvm.x86.avx512.fpclass.ps.512
  return _mm512_fpclass_ps_mask(__A, 4); 
}

__mmask8 test_mm_fpclass_sd_mask(__m128d __A)  { 
  // CHECK-LABEL: test_mm_fpclass_sd_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.sd
 return _mm_fpclass_sd_mask (__A, 2);
}

__mmask8 test_mm_mask_fpclass_sd_mask(__mmask8 __U, __m128d __A)  {
 // CHECK-LABEL: test_mm_mask_fpclass_sd_mask
 // CHECK: @llvm.x86.avx512.mask.fpclass.sd
 return _mm_mask_fpclass_sd_mask (__U,  __A, 2);
}

__mmask8 test_mm_fpclass_ss_mask(__m128 __A)  { 
 // CHECK-LABEL: test_mm_fpclass_ss_mask
 // CHECK: @llvm.x86.avx512.mask.fpclass.ss
 return _mm_fpclass_ss_mask ( __A, 2);
}

__mmask8 test_mm_mask_fpclass_ss_mask(__mmask8 __U, __m128 __A)  {
 // CHECK-LABEL: test_mm_mask_fpclass_ss_mask
 // CHECK: @llvm.x86.avx512.mask.fpclass.ss
 return _mm_mask_fpclass_ss_mask (__U, __A, 2);
}

