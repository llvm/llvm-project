// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s


#include <immintrin.h>
#include "builtin_test_helpers.h"

__mmask32 test_knot_mask32(__mmask32 a) {
  // CHECK-LABEL: @test_knot_mask32
  // CHECK: [[IN:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[NOT:%.*]] = xor <32 x i1> [[IN]], splat (i1 true)
  return _knot_mask32(a);
}

__mmask64 test_knot_mask64(__mmask64 a) {
  // CHECK-LABEL: @test_knot_mask64
  // CHECK: [[IN:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[NOT:%.*]] = xor <64 x i1> [[IN]], splat (i1 true)
  return _knot_mask64(a);
}

__mmask32 test_kand_mask32(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kand_mask32
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = and <32 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu16_mask(_kand_mask32(_mm512_cmpneq_epu16_mask(__A, __B),
                                                    _mm512_cmpneq_epu16_mask(__C, __D)),
                                                    __E, __F);
}

__mmask64 test_kand_mask64(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kand_mask64
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = and <64 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu8_mask(_kand_mask64(_mm512_cmpneq_epu8_mask(__A, __B),
                                                   _mm512_cmpneq_epu8_mask(__C, __D)),
                                                   __E, __F);
}

__mmask32 test_kandn_mask32(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kandn_mask32
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[NOT:%.*]] = xor <32 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = and <32 x i1> [[NOT]], [[RHS]]
  return _mm512_mask_cmpneq_epu16_mask(_kandn_mask32(_mm512_cmpneq_epu16_mask(__A, __B),
                                                     _mm512_cmpneq_epu16_mask(__C, __D)),
                                                     __E, __F);
}

__mmask64 test_kandn_mask64(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kandn_mask64
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[NOT:%.*]] = xor <64 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = and <64 x i1> [[NOT]], [[RHS]]
  return _mm512_mask_cmpneq_epu8_mask(_kandn_mask64(_mm512_cmpneq_epu8_mask(__A, __B),
                                                    _mm512_cmpneq_epu8_mask(__C, __D)),
                                                    __E, __F);
}

__mmask32 test_kor_mask32(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kor_mask32
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = or <32 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu16_mask(_kor_mask32(_mm512_cmpneq_epu16_mask(__A, __B),
                                                   _mm512_cmpneq_epu16_mask(__C, __D)),
                                                   __E, __F);
}

__mmask64 test_kor_mask64(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kor_mask64
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = or <64 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu8_mask(_kor_mask64(_mm512_cmpneq_epu8_mask(__A, __B),
                                                  _mm512_cmpneq_epu8_mask(__C, __D)),
                                                  __E, __F);
}

__mmask32 test_kxnor_mask32(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kxnor_mask32
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[NOT:%.*]] = xor <32 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = xor <32 x i1> [[NOT]], [[RHS]]
  return _mm512_mask_cmpneq_epu16_mask(_kxnor_mask32(_mm512_cmpneq_epu16_mask(__A, __B),
                                                     _mm512_cmpneq_epu16_mask(__C, __D)),
                                                     __E, __F);
}

__mmask64 test_kxnor_mask64(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kxnor_mask64
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[NOT:%.*]] = xor <64 x i1> [[LHS]], splat (i1 true)
  // CHECK: [[RES:%.*]] = xor <64 x i1> [[NOT]], [[RHS]]
  return _mm512_mask_cmpneq_epu8_mask(_kxnor_mask64(_mm512_cmpneq_epu8_mask(__A, __B),
                                                    _mm512_cmpneq_epu8_mask(__C, __D)),
                                                    __E, __F);
}

__mmask32 test_kxor_mask32(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kxor_mask32
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = xor <32 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu16_mask(_kxor_mask32(_mm512_cmpneq_epu16_mask(__A, __B),
                                                    _mm512_cmpneq_epu16_mask(__C, __D)),
                                                    __E, __F);
}

__mmask64 test_kxor_mask64(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kxor_mask64
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = xor <64 x i1> [[LHS]], [[RHS]]
  return _mm512_mask_cmpneq_epu8_mask(_kxor_mask64(_mm512_cmpneq_epu8_mask(__A, __B),
                                                   _mm512_cmpneq_epu8_mask(__C, __D)),
                                                   __E, __F);
}

unsigned char test_kortestz_mask32_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: @test_kortestz_mask32_u8
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[OR:%.*]] = or <32 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <32 x i1> [[OR]] to i32
  // CHECK: [[CMP:%.*]] = icmp eq i32 [[CAST]], 0
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  return _kortestz_mask32_u8(_mm512_cmpneq_epu16_mask(__A, __B),
                             _mm512_cmpneq_epu16_mask(__C, __D));
}

unsigned char test_kortestc_mask32_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: @test_kortestc_mask32_u8
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[OR:%.*]] = or <32 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <32 x i1> [[OR]] to i32
  // CHECK: [[CMP:%.*]] = icmp eq i32 [[CAST]], -1
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  return _kortestc_mask32_u8(_mm512_cmpneq_epu16_mask(__A, __B),
                             _mm512_cmpneq_epu16_mask(__C, __D));
}

unsigned char test_kortest_mask32_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, unsigned char *CF) {
  // CHECK-LABEL: @test_kortest_mask32_u8
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[OR:%.*]] = or <32 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <32 x i1> [[OR]] to i32
  // CHECK: [[CMP:%.*]] = icmp eq i32 [[CAST]], -1
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  // CHECK: [[LHS2:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS2:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[OR2:%.*]] = or <32 x i1> [[LHS2]], [[RHS2]]
  // CHECK: [[CAST2:%.*]] = bitcast <32 x i1> [[OR2]] to i32
  // CHECK: [[CMP2:%.*]] = icmp eq i32 [[CAST2]], 0
  // CHECK: [[ZEXT2:%.*]] = zext i1 [[CMP2]] to i32
  // CHECK: trunc i32 [[ZEXT2]] to i8
  return _kortest_mask32_u8(_mm512_cmpneq_epu16_mask(__A, __B),
                            _mm512_cmpneq_epu16_mask(__C, __D), CF);
}

unsigned char test_kortestz_mask64_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: @test_kortestz_mask64_u8
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[OR:%.*]] = or <64 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <64 x i1> [[OR]] to i64
  // CHECK: [[CMP:%.*]] = icmp eq i64 [[CAST]], 0
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  return _kortestz_mask64_u8(_mm512_cmpneq_epu8_mask(__A, __B),
                             _mm512_cmpneq_epu8_mask(__C, __D));
}

unsigned char test_kortestc_mask64_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: @test_kortestc_mask64_u8
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[OR:%.*]] = or <64 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <64 x i1> [[OR]] to i64
  // CHECK: [[CMP:%.*]] = icmp eq i64 [[CAST]], -1
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  return _kortestc_mask64_u8(_mm512_cmpneq_epu8_mask(__A, __B),
                             _mm512_cmpneq_epu8_mask(__C, __D));
}

unsigned char test_kortest_mask64_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, unsigned char *CF) {
  // CHECK-LABEL: @test_kortest_mask64_u8
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[OR:%.*]] = or <64 x i1> [[LHS]], [[RHS]]
  // CHECK: [[CAST:%.*]] = bitcast <64 x i1> [[OR]] to i64
  // CHECK: [[CMP:%.*]] = icmp eq i64 [[CAST]], -1
  // CHECK: [[ZEXT:%.*]] = zext i1 [[CMP]] to i32
  // CHECK: trunc i32 [[ZEXT]] to i8
  // CHECK: [[LHS2:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS2:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[OR2:%.*]] = or <64 x i1> [[LHS2]], [[RHS2]]
  // CHECK: [[CAST2:%.*]] = bitcast <64 x i1> [[OR2]] to i64
  // CHECK: [[CMP2:%.*]] = icmp eq i64 [[CAST2]], 0
  // CHECK: [[ZEXT2:%.*]] = zext i1 [[CMP2]] to i32
  // CHECK: trunc i32 [[ZEXT2]] to i8
  return _kortest_mask64_u8(_mm512_cmpneq_epu8_mask(__A, __B),
                            _mm512_cmpneq_epu8_mask(__C, __D), CF);
}

unsigned char test_ktestz_mask32_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: @test_ktestz_mask32_u8
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestz.d(<32 x i1> [[LHS]], <32 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktestz_mask32_u8(_mm512_cmpneq_epu16_mask(__A, __B),
                           _mm512_cmpneq_epu16_mask(__C, __D));
}

unsigned char test_ktestc_mask32_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: @test_ktestc_mask32_u8
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestc.d(<32 x i1> [[LHS]], <32 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktestc_mask32_u8(_mm512_cmpneq_epu16_mask(__A, __B),
                           _mm512_cmpneq_epu16_mask(__C, __D));
}

unsigned char test_ktest_mask32_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, unsigned char *CF) {
  // CHECK-LABEL: @test_ktest_mask32_u8
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestc.d(<32 x i1> [[LHS]], <32 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestz.d(<32 x i1> [[LHS]], <32 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktest_mask32_u8(_mm512_cmpneq_epu16_mask(__A, __B),
                          _mm512_cmpneq_epu16_mask(__C, __D), CF);
}

unsigned char test_ktestz_mask64_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: @test_ktestz_mask64_u8
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestz.q(<64 x i1> [[LHS]], <64 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktestz_mask64_u8(_mm512_cmpneq_epu8_mask(__A, __B),
                           _mm512_cmpneq_epu8_mask(__C, __D));
}

unsigned char test_ktestc_mask64_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D) {
  // CHECK-LABEL: @test_ktestc_mask64_u8
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestc.q(<64 x i1> [[LHS]], <64 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktestc_mask64_u8(_mm512_cmpneq_epu8_mask(__A, __B),
                           _mm512_cmpneq_epu8_mask(__C, __D));
}

unsigned char test_ktest_mask64_u8(__m512i __A, __m512i __B, __m512i __C, __m512i __D, unsigned char *CF) {
  // CHECK-LABEL: @test_ktest_mask64_u8
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestc.q(<64 x i1> [[LHS]], <64 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = call i32 @llvm.x86.avx512.ktestz.q(<64 x i1> [[LHS]], <64 x i1> [[RHS]])
  // CHECK: trunc i32 [[RES]] to i8
  return _ktest_mask64_u8(_mm512_cmpneq_epu8_mask(__A, __B),
                          _mm512_cmpneq_epu8_mask(__C, __D), CF);
}

__mmask32 test_kadd_mask32(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kadd_mask32
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = call <32 x i1> @llvm.x86.avx512.kadd.d(<32 x i1> [[LHS]], <32 x i1> [[RHS]])
  return _mm512_mask_cmpneq_epu16_mask(_kadd_mask32(_mm512_cmpneq_epu16_mask(__A, __B),
                                                    _mm512_cmpneq_epu16_mask(__C, __D)),
                                                    __E, __F);
}

__mmask64 test_kadd_mask64(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_kadd_mask64
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = call <64 x i1> @llvm.x86.avx512.kadd.q(<64 x i1> [[LHS]], <64 x i1> [[RHS]])
  return _mm512_mask_cmpneq_epu8_mask(_kadd_mask64(_mm512_cmpneq_epu8_mask(__A, __B),
                                                   _mm512_cmpneq_epu8_mask(__C, __D)),
                                                   __E, __F);
}

__mmask32 test_kshiftli_mask32(__m512i A, __m512i B, __m512i C, __m512i D) {
  // CHECK-LABEL: @test_kshiftli_mask32
  // CHECK: [[VAL:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = shufflevector <32 x i1> zeroinitializer, <32 x i1> [[VAL]], <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32>
  return _mm512_mask_cmpneq_epu16_mask(_kshiftli_mask32(_mm512_cmpneq_epu16_mask(A, B), 31), C, D);
}

__mmask32 test_kshiftri_mask32(__m512i A, __m512i B, __m512i C, __m512i D) {
  // CHECK-LABEL: @test_kshiftri_mask32
  // CHECK: [[VAL:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RES:%.*]] = shufflevector <32 x i1> [[VAL]], <32 x i1> zeroinitializer, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  return _mm512_mask_cmpneq_epu16_mask(_kshiftri_mask32(_mm512_cmpneq_epu16_mask(A, B), 31), C, D);
}

__mmask64 test_kshiftli_mask64(__m512i A, __m512i B, __m512i C, __m512i D) {
  // CHECK-LABEL: @test_kshiftli_mask64
  // CHECK: [[VAL:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = shufflevector <64 x i1> zeroinitializer, <64 x i1> [[VAL]], <64 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>
  return _mm512_mask_cmpneq_epu8_mask(_kshiftli_mask64(_mm512_cmpneq_epu8_mask(A, B), 32), C, D);
}

__mmask64 test_kshiftri_mask64(__m512i A, __m512i B, __m512i C, __m512i D) {
  // CHECK-LABEL: @test_kshiftri_mask64
  // CHECK: [[VAL:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RES:%.*]] = shufflevector <64 x i1> [[VAL]], <64 x i1> zeroinitializer, <64 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>
  return _mm512_mask_cmpneq_epu8_mask(_kshiftri_mask64(_mm512_cmpneq_epu8_mask(A, B), 32), C, D);
}

unsigned int test_cvtmask32_u32(__m512i A, __m512i B) {
  // CHECK-LABEL: @test_cvtmask32_u32
  return _cvtmask32_u32(_mm512_cmpneq_epu16_mask(A, B));
}

unsigned long long test_cvtmask64_u64(__m512i A, __m512i B) {
  // CHECK-LABEL: @test_cvtmask64_u64
  return _cvtmask64_u64(_mm512_cmpneq_epu8_mask(A, B));
}

__mmask32 test_cvtu32_mask32(__m512i A, __m512i B, unsigned int C) {
  // CHECK-LABEL: @test_cvtu32_mask32
  return _mm512_mask_cmpneq_epu16_mask(_cvtu32_mask32(C), A, B);
}

__mmask64 test_cvtu64_mask64(__m512i A, __m512i B, unsigned long long C) {
  // CHECK-LABEL: @test_cvtu64_mask64
  return _mm512_mask_cmpneq_epu8_mask(_cvtu64_mask64(C), A, B);
}

__mmask32 test_load_mask32(__mmask32 *A, __m512i B, __m512i C) {
  // CHECK-LABEL: @test_load_mask32
  // CHECK: [[LOAD:%.*]] = load i32, ptr %{{.*}}
  return _mm512_mask_cmpneq_epu16_mask(_load_mask32(A), B, C);
}

__mmask64 test_load_mask64(__mmask64 *A, __m512i B, __m512i C) {
  // CHECK-LABEL: @test_load_mask64
  // CHECK: [[LOAD:%.*]] = load i64, ptr %{{.*}}
  return _mm512_mask_cmpneq_epu8_mask(_load_mask64(A), B, C);
}

void test_store_mask32(__mmask32 *A, __m512i B, __m512i C) {
  // CHECK-LABEL: @test_store_mask32
  // CHECK: store i32 %{{.*}}, ptr %{{.*}}
  _store_mask32(A, _mm512_cmpneq_epu16_mask(B, C));
}

void test_store_mask64(__mmask64 *A, __m512i B, __m512i C) {
  // CHECK-LABEL: @test_store_mask64
  // CHECK: store i64 %{{.*}}, ptr %{{.*}}
  _store_mask64(A, _mm512_cmpneq_epu8_mask(B, C));
}

__mmask64 test_mm512_cmpeq_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi8_mask
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmpeq_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpeq_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi8_mask
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpeq_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi16_mask
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmpeq_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpeq_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi16_mask
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpgt_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epi8_mask
  // CHECK: icmp sgt <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmpgt_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpgt_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epi8_mask
  // CHECK: icmp sgt <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpgt_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epi16_mask
  // CHECK: icmp sgt <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmpgt_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpgt_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epi16_mask
  // CHECK: icmp sgt <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpeq_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epu8_mask
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmpeq_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpeq_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epu8_mask
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpeq_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epu16_mask
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmpeq_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpeq_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epu16_mask
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpgt_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epu8_mask
  // CHECK: icmp ugt <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmpgt_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpgt_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epu8_mask
  // CHECK: icmp ugt <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpgt_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epu16_mask
  // CHECK: icmp ugt <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmpgt_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpgt_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epu16_mask
  // CHECK: icmp ugt <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpge_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epi8_mask
  // CHECK: icmp sge <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmpge_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpge_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epi8_mask
  // CHECK: icmp sge <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpge_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epu8_mask
  // CHECK: icmp uge <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmpge_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpge_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epu8_mask
  // CHECK: icmp uge <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpge_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epi16_mask
  // CHECK: icmp sge <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmpge_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpge_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epi16_mask
  // CHECK: icmp sge <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpge_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epu16_mask
  // CHECK: icmp uge <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmpge_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpge_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epu16_mask
  // CHECK: icmp uge <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmple_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epi8_mask
  // CHECK: icmp sle <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmple_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmple_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epi8_mask
  // CHECK: icmp sle <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmple_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epu8_mask
  // CHECK: icmp ule <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmple_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmple_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epu8_mask
  // CHECK: icmp ule <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmple_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epi16_mask
  // CHECK: icmp sle <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmple_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmple_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epi16_mask
  // CHECK: icmp sle <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmple_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epu16_mask
  // CHECK: icmp ule <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmple_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmple_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epu16_mask
  // CHECK: icmp ule <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmplt_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epi8_mask
  // CHECK: icmp slt <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmplt_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmplt_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epi8_mask
  // CHECK: icmp slt <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmplt_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epu8_mask
  // CHECK: icmp ult <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmplt_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmplt_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epu8_mask
  // CHECK: icmp ult <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmplt_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epi16_mask
  // CHECK: icmp slt <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmplt_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmplt_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epi16_mask
  // CHECK: icmp slt <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmplt_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epu16_mask
  // CHECK: icmp ult <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmplt_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmplt_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epu16_mask
  // CHECK: icmp ult <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpneq_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epi8_mask
  // CHECK: icmp ne <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmpneq_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpneq_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epi8_mask
  // CHECK: icmp ne <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmpneq_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epu8_mask
  // CHECK: icmp ne <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmpneq_epu8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpneq_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epu8_mask
  // CHECK: icmp ne <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpneq_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epi16_mask
  // CHECK: icmp ne <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmpneq_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpneq_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epi16_mask
  // CHECK: icmp ne <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpneq_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epu16_mask
  // CHECK: icmp ne <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmpneq_epu16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpneq_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epu16_mask
  // CHECK: icmp ne <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask64 test_mm512_cmp_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epi8_mask
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmp_epi8_mask(__a, __b, 0);
}

__mmask64 test_mm512_mask_cmp_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epi8_mask
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmp_epi8_mask(__u, __a, __b, 0);
}

__mmask64 test_mm512_cmp_epu8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epu8_mask
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_cmp_epu8_mask(__a, __b, 0);
}

__mmask64 test_mm512_mask_cmp_epu8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epu8_mask
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return (__mmask64)_mm512_mask_cmp_epu8_mask(__u, __a, __b, 0);
}

__mmask32 test_mm512_cmp_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epi16_mask
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmp_epi16_mask(__a, __b, 0);
}

__mmask32 test_mm512_mask_cmp_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epi16_mask
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmp_epi16_mask(__u, __a, __b, 0);
}

__mmask32 test_mm512_cmp_epu16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epu16_mask
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_cmp_epu16_mask(__a, __b, 0);
}

__mmask32 test_mm512_mask_cmp_epu16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epu16_mask
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm512_mask_cmp_epu16_mask(__u, __a, __b, 0);
}

__m512i test_mm512_add_epi8 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_add_epi8
  //CHECK: add <64 x i8>
  return _mm512_add_epi8(__A,__B);
}

__m512i test_mm512_mask_add_epi8 (__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_add_epi8
  //CHECK: add <64 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_add_epi8(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_add_epi8 (__mmask64 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_add_epi8
  //CHECK: add <64 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_add_epi8(__U, __A, __B);
}

__m512i test_mm512_sub_epi8 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_sub_epi8
  //CHECK: sub <64 x i8>
  return _mm512_sub_epi8(__A, __B);
}

__m512i test_mm512_mask_sub_epi8 (__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_sub_epi8
  //CHECK: sub <64 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_sub_epi8(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_sub_epi8 (__mmask64 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_sub_epi8
  //CHECK: sub <64 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_sub_epi8(__U, __A, __B);
}

__m512i test_mm512_add_epi16 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_add_epi16
  //CHECK: add <32 x i16>
  return _mm512_add_epi16(__A, __B);
}

__m512i test_mm512_mask_add_epi16 (__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_add_epi16
  //CHECK: add <32 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_add_epi16(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_add_epi16 (__mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_add_epi16
  //CHECK: add <32 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_add_epi16(__U, __A, __B);
}

__m512i test_mm512_sub_epi16 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_sub_epi16
  //CHECK: sub <32 x i16>
  return _mm512_sub_epi16(__A, __B);
}

__m512i test_mm512_mask_sub_epi16 (__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_sub_epi16
  //CHECK: sub <32 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_sub_epi16(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_sub_epi16 (__mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_sub_epi16
  //CHECK: sub <32 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_sub_epi16(__U, __A, __B);
}

__m512i test_mm512_mullo_epi16 (__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mullo_epi16
  //CHECK: mul <32 x i16>
  return _mm512_mullo_epi16(__A, __B);
}
TEST_CONSTEXPR(match_v32hi(_mm512_mullo_epi16((__m512i)(__v32hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m512i)(__v32hi){-64, -62, +60, +58, -56, -54, +52, +50, -48, -46, +44, +42, -40, -38, +36, +34, -32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), -64, 124, 180, -232, -280, 324, 364, -400, -432, 460, 484, -504, -520, 532, 540, -544, -544, 540, 532, -520, -504, 484, 460, -432, -400, 364, 324, -280, -232, -180, -124, -64));

__m512i test_mm512_mask_mullo_epi16 (__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mask_mullo_epi16
  //CHECK: mul <32 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_mullo_epi16(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_mullo_epi16 (__mmask32 __U, __m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_mullo_epi16
  //CHECK: mul <32 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_mullo_epi16(__U, __A, __B);
}

__m512i test_mm512_mask_blend_epi8(__mmask64 __U, __m512i __A, __m512i __W) {
  // CHECK-LABEL: @test_mm512_mask_blend_epi8
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_blend_epi8(__U,__A,__W); 
}
__m512i test_mm512_mask_blend_epi16(__mmask32 __U, __m512i __A, __m512i __W) {
  // CHECK-LABEL: @test_mm512_mask_blend_epi16
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_blend_epi16(__U,__A,__W); 
}
__m512i test_mm512_abs_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_abs_epi8
  // CHECK: [[ABS:%.*]] = call <64 x i8> @llvm.abs.v64i8(<64 x i8> %{{.*}}, i1 false)
  return _mm512_abs_epi8(__A); 
}
__m512i test_mm512_mask_abs_epi8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_abs_epi8
  // CHECK: [[ABS:%.*]] = call <64 x i8> @llvm.abs.v64i8(<64 x i8> %{{.*}}, i1 false)
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> [[ABS]], <64 x i8> %{{.*}}
  return _mm512_mask_abs_epi8(__W,__U,__A); 
}
__m512i test_mm512_maskz_abs_epi8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_abs_epi8
  // CHECK: [[ABS:%.*]] = call <64 x i8> @llvm.abs.v64i8(<64 x i8> %{{.*}}, i1 false)
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> [[ABS]], <64 x i8> %{{.*}}
  return _mm512_maskz_abs_epi8(__U,__A); 
}
__m512i test_mm512_abs_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_abs_epi16
  // CHECK: [[ABS:%.*]] = call <32 x i16> @llvm.abs.v32i16(<32 x i16> %{{.*}}, i1 false)
  return _mm512_abs_epi16(__A); 
}
__m512i test_mm512_mask_abs_epi16(__m512i __W, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_abs_epi16
  // CHECK: [[ABS:%.*]] = call <32 x i16> @llvm.abs.v32i16(<32 x i16> %{{.*}}, i1 false)
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> [[ABS]], <32 x i16> %{{.*}}
  return _mm512_mask_abs_epi16(__W,__U,__A); 
}
__m512i test_mm512_maskz_abs_epi16(__mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_abs_epi16
  // CHECK: [[ABS:%.*]] = call <32 x i16> @llvm.abs.v32i16(<32 x i16> %{{.*}}, i1 false)
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> [[ABS]], <32 x i16> %{{.*}}
  return _mm512_maskz_abs_epi16(__U,__A); 
}
__m512i test_mm512_packs_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_packs_epi32
  // CHECK: @llvm.x86.avx512.packssdw.512
  return _mm512_packs_epi32(__A,__B); 
}
__m512i test_mm512_maskz_packs_epi32(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_packs_epi32
  // CHECK: @llvm.x86.avx512.packssdw.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_packs_epi32(__M,__A,__B); 
}
__m512i test_mm512_mask_packs_epi32(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_packs_epi32
  // CHECK: @llvm.x86.avx512.packssdw.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_packs_epi32(__W,__M,__A,__B); 
}
__m512i test_mm512_packs_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_packs_epi16
  // CHECK: @llvm.x86.avx512.packsswb.512
  return _mm512_packs_epi16(__A,__B); 
}
__m512i test_mm512_mask_packs_epi16(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_packs_epi16
  // CHECK: @llvm.x86.avx512.packsswb.512
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_packs_epi16(__W,__M,__A,__B); 
}
__m512i test_mm512_maskz_packs_epi16(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_packs_epi16
  // CHECK: @llvm.x86.avx512.packsswb.512
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_packs_epi16(__M,__A,__B); 
}
__m512i test_mm512_packus_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_packus_epi32
  // CHECK: @llvm.x86.avx512.packusdw.512
  return _mm512_packus_epi32(__A,__B); 
}
__m512i test_mm512_maskz_packus_epi32(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_packus_epi32
  // CHECK: @llvm.x86.avx512.packusdw.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_packus_epi32(__M,__A,__B); 
}
__m512i test_mm512_mask_packus_epi32(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_packus_epi32
  // CHECK: @llvm.x86.avx512.packusdw.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_packus_epi32(__W,__M,__A,__B); 
}
__m512i test_mm512_packus_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_packus_epi16
  // CHECK: @llvm.x86.avx512.packuswb.512
  return _mm512_packus_epi16(__A,__B); 
}
__m512i test_mm512_mask_packus_epi16(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_packus_epi16
  // CHECK: @llvm.x86.avx512.packuswb.512
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_packus_epi16(__W,__M,__A,__B); 
}
__m512i test_mm512_maskz_packus_epi16(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_packus_epi16
  // CHECK: @llvm.x86.avx512.packuswb.512
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_packus_epi16(__M,__A,__B); 
}
__m512i test_mm512_adds_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_adds_epi8
  // CHECK: @llvm.sadd.sat.v64i8
  return _mm512_adds_epi8(__A,__B); 
}
__m512i test_mm512_mask_adds_epi8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_adds_epi8
  // CHECK: @llvm.sadd.sat.v64i8
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
 return _mm512_mask_adds_epi8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_adds_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_adds_epi8
  // CHECK: @llvm.sadd.sat.v64i8
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_adds_epi8(__U,__A,__B); 
}
__m512i test_mm512_adds_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_adds_epi16
  // CHECK: @llvm.sadd.sat.v32i16
 return _mm512_adds_epi16(__A,__B); 
}
__m512i test_mm512_mask_adds_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_adds_epi16
  // CHECK: @llvm.sadd.sat.v32i16
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_adds_epi16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_adds_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_adds_epi16
  // CHECK: @llvm.sadd.sat.v32i16
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
return _mm512_maskz_adds_epi16(__U,__A,__B); 
}
__m512i test_mm512_adds_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_adds_epu8
  // CHECK-NOT: @llvm.x86.avx512.mask.paddus.b.512
  // CHECK: call <64 x i8> @llvm.uadd.sat.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_adds_epu8(__A,__B); 
}
__m512i test_mm512_mask_adds_epu8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_adds_epu8
  // CHECK-NOT: @llvm.x86.avx512.mask.paddus.b.512
  // CHECK: call <64 x i8> @llvm.uadd.sat.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_adds_epu8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_adds_epu8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_adds_epu8
  // CHECK-NOT: @llvm.x86.avx512.mask.paddus.b.512
  // CHECK: call <64 x i8> @llvm.uadd.sat.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_adds_epu8(__U,__A,__B); 
}
__m512i test_mm512_adds_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_adds_epu16
  // CHECK-NOT: @llvm.x86.avx512.mask.paddus.w.512
  // CHECK: call <32 x i16> @llvm.uadd.sat.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_adds_epu16(__A,__B); 
}
__m512i test_mm512_mask_adds_epu16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_adds_epu16
  // CHECK-NOT: @llvm.x86.avx512.mask.paddus.w.512
  // CHECK: call <32 x i16> @llvm.uadd.sat.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_adds_epu16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_adds_epu16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_adds_epu16
  // CHECK-NOT: @llvm.x86.avx512.mask.paddus.w.512
  // CHECK: call <32 x i16> @llvm.uadd.sat.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_adds_epu16(__U,__A,__B); 
}
__m512i test_mm512_avg_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_avg_epu8
  // CHECK: @llvm.x86.avx512.pavg.b.512
  return _mm512_avg_epu8(__A,__B); 
}
__m512i test_mm512_mask_avg_epu8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_avg_epu8
  // CHECK: @llvm.x86.avx512.pavg.b.512
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_avg_epu8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_avg_epu8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_avg_epu8
  // CHECK: @llvm.x86.avx512.pavg.b.512
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_avg_epu8(__U,__A,__B); 
}
__m512i test_mm512_avg_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_avg_epu16
  // CHECK: @llvm.x86.avx512.pavg.w.512
  return _mm512_avg_epu16(__A,__B); 
}
__m512i test_mm512_mask_avg_epu16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_avg_epu16
  // CHECK: @llvm.x86.avx512.pavg.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_avg_epu16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_avg_epu16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_avg_epu16
  // CHECK: @llvm.x86.avx512.pavg.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_avg_epu16(__U,__A,__B); 
}
__m512i test_mm512_max_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_max_epi8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.smax.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_max_epi8(__A,__B); 
}
__m512i test_mm512_maskz_max_epi8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_max_epi8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.smax.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK:       select <64 x i1> {{.*}}, <64 x i8> [[RES]], <64 x i8> {{.*}}
  return _mm512_maskz_max_epi8(__M,__A,__B); 
}
__m512i test_mm512_mask_max_epi8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_max_epi8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.smax.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK:       select <64 x i1> {{.*}}, <64 x i8> [[RES]], <64 x i8> {{.*}}
  return _mm512_mask_max_epi8(__W,__M,__A,__B); 
}
__m512i test_mm512_max_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_max_epi16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.smax.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_max_epi16(__A,__B); 
}
__m512i test_mm512_maskz_max_epi16(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_max_epi16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.smax.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:       select <32 x i1> {{.*}}, <32 x i16> [[RES]], <32 x i16> {{.*}}
  return _mm512_maskz_max_epi16(__M,__A,__B); 
}
__m512i test_mm512_mask_max_epi16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_max_epi16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.smax.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:       select <32 x i1> {{.*}}, <32 x i16> [[RES]], <32 x i16> {{.*}}
  return _mm512_mask_max_epi16(__W,__M,__A,__B); 
}
__m512i test_mm512_max_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_max_epu8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.umax.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_max_epu8(__A,__B); 
}
__m512i test_mm512_maskz_max_epu8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_max_epu8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.umax.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK:       select <64 x i1> {{.*}}, <64 x i8> [[RES]], <64 x i8> {{.*}}
  return _mm512_maskz_max_epu8(__M,__A,__B); 
}
__m512i test_mm512_mask_max_epu8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_max_epu8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.umax.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK:       select <64 x i1> {{.*}}, <64 x i8> [[RES]], <64 x i8> {{.*}}
  return _mm512_mask_max_epu8(__W,__M,__A,__B); 
}
__m512i test_mm512_max_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_max_epu16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.umax.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_max_epu16(__A,__B); 
}
__m512i test_mm512_maskz_max_epu16(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_max_epu16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.umax.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:       select <32 x i1> {{.*}}, <32 x i16> [[RES]], <32 x i16> {{.*}}
  return _mm512_maskz_max_epu16(__M,__A,__B); 
}
__m512i test_mm512_mask_max_epu16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_max_epu16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.umax.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:       select <32 x i1> {{.*}}, <32 x i16> [[RES]], <32 x i16> {{.*}}
  return _mm512_mask_max_epu16(__W,__M,__A,__B); 
}
__m512i test_mm512_min_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_min_epi8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.smin.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_min_epi8(__A,__B); 
}
__m512i test_mm512_maskz_min_epi8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_min_epi8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.smin.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK:       select <64 x i1> {{.*}}, <64 x i8> [[RES]], <64 x i8> {{.*}}
  return _mm512_maskz_min_epi8(__M,__A,__B); 
}
__m512i test_mm512_mask_min_epi8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_min_epi8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.smin.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK:       select <64 x i1> {{.*}}, <64 x i8> [[RES]], <64 x i8> {{.*}}
  return _mm512_mask_min_epi8(__W,__M,__A,__B); 
}
__m512i test_mm512_min_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_min_epi16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.smin.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_min_epi16(__A,__B); 
}
__m512i test_mm512_maskz_min_epi16(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_min_epi16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.smin.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:       select <32 x i1> {{.*}}, <32 x i16> [[RES]], <32 x i16> {{.*}}
  return _mm512_maskz_min_epi16(__M,__A,__B); 
}
__m512i test_mm512_mask_min_epi16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_min_epi16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.smin.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:       select <32 x i1> {{.*}}, <32 x i16> [[RES]], <32 x i16> {{.*}}
  return _mm512_mask_min_epi16(__W,__M,__A,__B); 
}
__m512i test_mm512_min_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_min_epu8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.umin.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_min_epu8(__A,__B); 
}
__m512i test_mm512_maskz_min_epu8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_min_epu8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.umin.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK:       select <64 x i1> {{.*}}, <64 x i8> [[RES]], <64 x i8> {{.*}}
  return _mm512_maskz_min_epu8(__M,__A,__B); 
}
__m512i test_mm512_mask_min_epu8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_min_epu8
  // CHECK:       [[RES:%.*]] = call <64 x i8> @llvm.umin.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK:       select <64 x i1> {{.*}}, <64 x i8> [[RES]], <64 x i8> {{.*}}
  return _mm512_mask_min_epu8(__W,__M,__A,__B); 
}
__m512i test_mm512_min_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_min_epu16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.umin.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_min_epu16(__A,__B); 
}
__m512i test_mm512_maskz_min_epu16(__mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_min_epu16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.umin.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:       select <32 x i1> {{.*}}, <32 x i16> [[RES]], <32 x i16> {{.*}}
  return _mm512_maskz_min_epu16(__M,__A,__B); 
}
__m512i test_mm512_mask_min_epu16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_min_epu16
  // CHECK:       [[RES:%.*]] = call <32 x i16> @llvm.umin.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:       select <32 x i1> {{.*}}, <32 x i16> [[RES]], <32 x i16> {{.*}}
  return _mm512_mask_min_epu16(__W,__M,__A,__B); 
}
__m512i test_mm512_shuffle_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shuffle_epi8
  // CHECK: @llvm.x86.avx512.pshuf.b.512
  return _mm512_shuffle_epi8(__A,__B); 
}
__m512i test_mm512_mask_shuffle_epi8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shuffle_epi8
  // CHECK: @llvm.x86.avx512.pshuf.b.512
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_shuffle_epi8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_shuffle_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shuffle_epi8
  // CHECK: @llvm.x86.avx512.pshuf.b.512
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_shuffle_epi8(__U,__A,__B); 
}
__m512i test_mm512_subs_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_subs_epi8
  // CHECK: @llvm.ssub.sat.v64i8
return _mm512_subs_epi8(__A,__B); 
}
__m512i test_mm512_mask_subs_epi8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_subs_epi8
  // CHECK: @llvm.ssub.sat.v64i8
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
return _mm512_mask_subs_epi8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_subs_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_subs_epi8
  // CHECK: @llvm.ssub.sat.v64i8
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
return _mm512_maskz_subs_epi8(__U,__A,__B); 
}
__m512i test_mm512_subs_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_subs_epi16
  // CHECK: @llvm.ssub.sat.v32i16
return _mm512_subs_epi16(__A,__B); 
}
__m512i test_mm512_mask_subs_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_subs_epi16
  // CHECK: @llvm.ssub.sat.v32i16
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
return _mm512_mask_subs_epi16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_subs_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_subs_epi16
  // CHECK: @llvm.ssub.sat.v32i16
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
return _mm512_maskz_subs_epi16(__U,__A,__B); 
}
__m512i test_mm512_subs_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_subs_epu8
  // CHECK-NOT: @llvm.x86.avx512.mask.psubus.b.512
  // CHECK: call <64 x i8> @llvm.usub.sat.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
return _mm512_subs_epu8(__A,__B); 
}
__m512i test_mm512_mask_subs_epu8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_subs_epu8
  // CHECK-NOT: @llvm.x86.avx512.mask.psubus.b.512
  // CHECK: call <64 x i8> @llvm.usub.sat.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
return _mm512_mask_subs_epu8(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_subs_epu8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_subs_epu8
  // CHECK-NOT: @llvm.x86.avx512.mask.psubus.b.512
  // CHECK: call <64 x i8> @llvm.usub.sat.v64i8(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
return _mm512_maskz_subs_epu8(__U,__A,__B); 
}
__m512i test_mm512_subs_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_subs_epu16
  // CHECK-NOT: @llvm.x86.avx512.mask.psubus.w.512
  // CHECK: call <32 x i16> @llvm.usub.sat.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
return _mm512_subs_epu16(__A,__B); 
}
__m512i test_mm512_mask_subs_epu16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_subs_epu16
  // CHECK-NOT: @llvm.x86.avx512.mask.psubus.w.512
  // CHECK: call <32 x i16> @llvm.usub.sat.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
return _mm512_mask_subs_epu16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_subs_epu16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_subs_epu16
  // CHECK-NOT: @llvm.x86.avx512.mask.psubus.w.512
  // CHECK: call <32 x i16> @llvm.usub.sat.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
return _mm512_maskz_subs_epu16(__U,__A,__B); 
}
__m512i test_mm512_mask2_permutex2var_epi16(__m512i __A, __m512i __I, __mmask32 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask2_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask2_permutex2var_epi16(__A,__I,__U,__B); 
}
__m512i test_mm512_permutex2var_epi16(__m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.512
  return _mm512_permutex2var_epi16(__A,__I,__B); 
}
__m512i test_mm512_mask_permutex2var_epi16(__m512i __A, __mmask32 __U, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_permutex2var_epi16(__A,__U,__I,__B); 
}
__m512i test_mm512_maskz_permutex2var_epi16(__mmask32 __U, __m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_permutex2var_epi16(__U,__A,__I,__B); 
}

__m512i test_mm512_mulhrs_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mulhrs_epi16
  // CHECK: @llvm.x86.avx512.pmul.hr.sw.512
  return _mm512_mulhrs_epi16(__A,__B); 
}
__m512i test_mm512_mask_mulhrs_epi16(__m512i __W, __mmask32 __U, __m512i __A,        __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_mulhrs_epi16
  // CHECK: @llvm.x86.avx512.pmul.hr.sw.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_mulhrs_epi16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_mulhrs_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.avx512.pmul.hr.sw.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_mulhrs_epi16(__U,__A,__B); 
}
__m512i test_mm512_mulhi_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mulhi_epi16
  // CHECK: @llvm.x86.avx512.pmulh.w.512
  return _mm512_mulhi_epi16(__A,__B); 
}
TEST_CONSTEXPR(match_v32hi(_mm512_mulhi_epi16((__m512i)(__v32hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m512i)(__v32hi){-64, -62, +60, +58, -56, -54, +52, +50, -48, -46, +44, +42, -40, -38, +36, +34, -32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1));

__m512i test_mm512_mask_mulhi_epi16(__m512i __W, __mmask32 __U, __m512i __A,       __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_mulhi_epi16
  // CHECK: @llvm.x86.avx512.pmulh.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_mulhi_epi16(__W,__U,__A,__B); 
}

__m512i test_mm512_maskz_mulhi_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_mulhi_epi16
  // CHECK: @llvm.x86.avx512.pmulh.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_mulhi_epi16(__U,__A,__B); 
}

__m512i test_mm512_mulhi_epu16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mulhi_epu16
  // CHECK: @llvm.x86.avx512.pmulhu.w.512
  return _mm512_mulhi_epu16(__A,__B); 
}
TEST_CONSTEXPR(match_v32hi(_mm512_mulhi_epu16((__m512i)(__v32hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m512i)(__v32hi){-64, -62, +60, +58, -56, -54, +52, +50, -48, -46, +44, +42, -40, -38, +36, +34, -32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), 0, -64, 0, 57, 4, -60, 0, 49, 8, -56, 0, 41, 12, -52, 0, 33, 16, -48, 0, 25, 20, -44, 0, 17, 24, -40, 0, 9, 28, 5, 30, 1));

__m512i test_mm512_mask_mulhi_epu16(__m512i __W, __mmask32 __U, __m512i __A,       __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_mulhi_epu16
  // CHECK: @llvm.x86.avx512.pmulhu.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_mulhi_epu16(__W,__U,__A,__B); 
}

__m512i test_mm512_maskz_mulhi_epu16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_mulhi_epu16
  // CHECK: @llvm.x86.avx512.pmulhu.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_mulhi_epu16(__U,__A,__B); 
}

__m512i test_mm512_maddubs_epi16(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maddubs_epi16
  // CHECK: @llvm.x86.avx512.pmaddubs.w.512
  return _mm512_maddubs_epi16(__X,__Y); 
}
__m512i test_mm512_mask_maddubs_epi16(__m512i __W, __mmask32 __U, __m512i __X,         __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_maddubs_epi16
  // CHECK: @llvm.x86.avx512.pmaddubs.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_maddubs_epi16(__W,__U,__X,__Y); 
}
__m512i test_mm512_maskz_maddubs_epi16(__mmask32 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maskz_maddubs_epi16
  // CHECK: @llvm.x86.avx512.pmaddubs.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_maddubs_epi16(__U,__X,__Y); 
}
__m512i test_mm512_madd_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_madd_epi16
  // CHECK: @llvm.x86.avx512.pmaddw.d.512
  return _mm512_madd_epi16(__A,__B); 
}
__m512i test_mm512_mask_madd_epi16(__m512i __W, __mmask16 __U, __m512i __A,      __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_madd_epi16
  // CHECK: @llvm.x86.avx512.pmaddw.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_madd_epi16(__W,__U,__A,__B); 
}
__m512i test_mm512_maskz_madd_epi16(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_madd_epi16
  // CHECK: @llvm.x86.avx512.pmaddw.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_madd_epi16(__U,__A,__B); 
}

__m256i test_mm512_cvtsepi16_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.512
  return _mm512_cvtsepi16_epi8(__A); 
}

__m256i test_mm512_mask_cvtsepi16_epi8(__m256i __O, __mmask32 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.512
  return _mm512_mask_cvtsepi16_epi8(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtsepi16_epi8(__mmask32 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.512
  return _mm512_maskz_cvtsepi16_epi8(__M, __A); 
}

__m256i test_mm512_cvtusepi16_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.512
  return _mm512_cvtusepi16_epi8(__A); 
}

__m256i test_mm512_mask_cvtusepi16_epi8(__m256i __O, __mmask32 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.512
  return _mm512_mask_cvtusepi16_epi8(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtusepi16_epi8(__mmask32 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.512
  return _mm512_maskz_cvtusepi16_epi8(__M, __A); 
}

__m256i test_mm512_cvtepi16_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi16_epi8
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  return _mm512_cvtepi16_epi8(__A); 
}

__m256i test_mm512_mask_cvtepi16_epi8(__m256i __O, __mmask32 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi16_epi8
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm512_mask_cvtepi16_epi8(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtepi16_epi8(__mmask32 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi16_epi8
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm512_maskz_cvtepi16_epi8(__M, __A); 
}

__m512i test_mm512_unpackhi_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_unpackhi_epi8
  // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 8, i32 72, i32 9, i32 73, i32 10, i32 74, i32 11, i32 75, i32 12, i32 76, i32 13, i32 77, i32 14, i32 78, i32 15, i32 79, i32 24, i32 88, i32 25, i32 89, i32 26, i32 90, i32 27, i32 91, i32 28, i32 92, i32 29, i32 93, i32 30, i32 94, i32 31, i32 95, i32 40, i32 104, i32 41, i32 105, i32 42, i32 106, i32 43, i32 107, i32 44, i32 108, i32 45, i32 109, i32 46, i32 110, i32 47, i32 111, i32 56, i32 120, i32 57, i32 121, i32 58, i32 122, i32 59, i32 123, i32 60, i32 124, i32 61, i32 125, i32 62, i32 126, i32 63, i32 127>
  return _mm512_unpackhi_epi8(__A, __B); 
}

__m512i test_mm512_mask_unpackhi_epi8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_unpackhi_epi8
  // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 8, i32 72, i32 9, i32 73, i32 10, i32 74, i32 11, i32 75, i32 12, i32 76, i32 13, i32 77, i32 14, i32 78, i32 15, i32 79, i32 24, i32 88, i32 25, i32 89, i32 26, i32 90, i32 27, i32 91, i32 28, i32 92, i32 29, i32 93, i32 30, i32 94, i32 31, i32 95, i32 40, i32 104, i32 41, i32 105, i32 42, i32 106, i32 43, i32 107, i32 44, i32 108, i32 45, i32 109, i32 46, i32 110, i32 47, i32 111, i32 56, i32 120, i32 57, i32 121, i32 58, i32 122, i32 59, i32 123, i32 60, i32 124, i32 61, i32 125, i32 62, i32 126, i32 63, i32 127>
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_unpackhi_epi8(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpackhi_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpackhi_epi8
  // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 8, i32 72, i32 9, i32 73, i32 10, i32 74, i32 11, i32 75, i32 12, i32 76, i32 13, i32 77, i32 14, i32 78, i32 15, i32 79, i32 24, i32 88, i32 25, i32 89, i32 26, i32 90, i32 27, i32 91, i32 28, i32 92, i32 29, i32 93, i32 30, i32 94, i32 31, i32 95, i32 40, i32 104, i32 41, i32 105, i32 42, i32 106, i32 43, i32 107, i32 44, i32 108, i32 45, i32 109, i32 46, i32 110, i32 47, i32 111, i32 56, i32 120, i32 57, i32 121, i32 58, i32 122, i32 59, i32 123, i32 60, i32 124, i32 61, i32 125, i32 62, i32 126, i32 63, i32 127>
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_unpackhi_epi8(__U, __A, __B); 
}

__m512i test_mm512_unpackhi_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_unpackhi_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i32> <i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  return _mm512_unpackhi_epi16(__A, __B); 
}

__m512i test_mm512_mask_unpackhi_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_unpackhi_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i32> <i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_unpackhi_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpackhi_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpackhi_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i32> <i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_unpackhi_epi16(__U, __A, __B); 
}

__m512i test_mm512_unpacklo_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_unpacklo_epi8
  // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 0, i32 64, i32 1, i32 65, i32 2, i32 66, i32 3, i32 67, i32 4, i32 68, i32 5, i32 69, i32 6, i32 70, i32 7, i32 71, i32 16, i32 80, i32 17, i32 81, i32 18, i32 82, i32 19, i32 83, i32 20, i32 84, i32 21, i32 85, i32 22, i32 86, i32 23, i32 87, i32 32, i32 96, i32 33, i32 97, i32 34, i32 98, i32 35, i32 99, i32 36, i32 100, i32 37, i32 101, i32 38, i32 102, i32 39, i32 103, i32 48, i32 112, i32 49, i32 113, i32 50, i32 114, i32 51, i32 115, i32 52, i32 116, i32 53, i32 117, i32 54, i32 118, i32 55, i32 119>
  return _mm512_unpacklo_epi8(__A, __B); 
}

__m512i test_mm512_mask_unpacklo_epi8(__m512i __W, __mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_unpacklo_epi8
  // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 0, i32 64, i32 1, i32 65, i32 2, i32 66, i32 3, i32 67, i32 4, i32 68, i32 5, i32 69, i32 6, i32 70, i32 7, i32 71, i32 16, i32 80, i32 17, i32 81, i32 18, i32 82, i32 19, i32 83, i32 20, i32 84, i32 21, i32 85, i32 22, i32 86, i32 23, i32 87, i32 32, i32 96, i32 33, i32 97, i32 34, i32 98, i32 35, i32 99, i32 36, i32 100, i32 37, i32 101, i32 38, i32 102, i32 39, i32 103, i32 48, i32 112, i32 49, i32 113, i32 50, i32 114, i32 51, i32 115, i32 52, i32 116, i32 53, i32 117, i32 54, i32 118, i32 55, i32 119>
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_unpacklo_epi8(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpacklo_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpacklo_epi8
  // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 0, i32 64, i32 1, i32 65, i32 2, i32 66, i32 3, i32 67, i32 4, i32 68, i32 5, i32 69, i32 6, i32 70, i32 7, i32 71, i32 16, i32 80, i32 17, i32 81, i32 18, i32 82, i32 19, i32 83, i32 20, i32 84, i32 21, i32 85, i32 22, i32 86, i32 23, i32 87, i32 32, i32 96, i32 33, i32 97, i32 34, i32 98, i32 35, i32 99, i32 36, i32 100, i32 37, i32 101, i32 38, i32 102, i32 39, i32 103, i32 48, i32 112, i32 49, i32 113, i32 50, i32 114, i32 51, i32 115, i32 52, i32 116, i32 53, i32 117, i32 54, i32 118, i32 55, i32 119>
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_unpacklo_epi8(__U, __A, __B); 
}

__m512i test_mm512_unpacklo_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_unpacklo_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59>
  return _mm512_unpacklo_epi16(__A, __B); 
}

__m512i test_mm512_mask_unpacklo_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_unpacklo_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_unpacklo_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpacklo_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpacklo_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_unpacklo_epi16(__U, __A, __B); 
}

__m512i test_mm512_cvtepi8_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi8_epi16
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  return _mm512_cvtepi8_epi16(__A); 
}

TEST_CONSTEXPR(match_v32hi(_mm512_cvtepi8_epi16(_mm256_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28)), -3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28));

__m512i test_mm512_mask_cvtepi8_epi16(__m512i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi8_epi16
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_cvtepi8_epi16(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi8_epi16(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi8_epi16
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_cvtepi8_epi16(__U, __A); 
}

__m512i test_mm512_cvtepu8_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm512_cvtepu8_epi16
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  return _mm512_cvtepu8_epi16(__A); 
}

TEST_CONSTEXPR(match_v32hi(_mm512_cvtepu8_epi16(_mm256_setr_epi8(-3, 2, -1, 0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28)), 253, 2, 255, 0, 1, 254, 3, 252, 5, 250, 7, 248, 9, 246, 11, 244, 13, 242, 15, 240, 17, 238, 19, 236, 21, 234, 23, 232, 25, 230, 27, 228));

__m512i test_mm512_mask_cvtepu8_epi16(__m512i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepu8_epi16
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_cvtepu8_epi16(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu8_epi16(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepu8_epi16
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_cvtepu8_epi16(__U, __A); 
}

__m512i test_mm512_shufflehi_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_shufflehi_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>
  return _mm512_shufflehi_epi16(__A, 5); 
}

__m512i test_mm512_mask_shufflehi_epi16(__m512i __W, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_shufflehi_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shufflehi_epi16(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_shufflehi_epi16(__mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_shufflehi_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shufflehi_epi16(__U, __A, 5); 
}

__m512i test_mm512_shufflelo_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_shufflelo_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>
  return _mm512_shufflelo_epi16(__A, 5); 
}

__m512i test_mm512_mask_shufflelo_epi16(__m512i __W, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_shufflelo_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shufflelo_epi16(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_shufflelo_epi16(__mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_shufflelo_epi16
  // CHECK: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shufflelo_epi16(__U, __A, 5); 
}

__m512i test_mm512_sllv_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.512(
  return _mm512_sllv_epi16(__A, __B); 
}

__m512i test_mm512_mask_sllv_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_sllv_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sllv_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_sllv_epi16(__U, __A, __B); 
}

__m512i test_mm512_sll_epi16(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_sll_epi16
  // CHECK: @llvm.x86.avx512.psll.w.512
  return _mm512_sll_epi16(__A, __B); 
}

__m512i test_mm512_mask_sll_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_sll_epi16
  // CHECK: @llvm.x86.avx512.psll.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_sll_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sll_epi16(__mmask32 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_sll_epi16
  // CHECK: @llvm.x86.avx512.psll.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_sll_epi16(__U, __A, __B); 
}

__m512i test_mm512_slli_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_slli_epi16
  // CHECK: @llvm.x86.avx512.pslli.w.512
  return _mm512_slli_epi16(__A, 5); 
}

__m512i test_mm512_slli_epi16_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm512_slli_epi16_2
  // CHECK: @llvm.x86.avx512.pslli.w.512
  return _mm512_slli_epi16(__A, __B); 
}

__m512i test_mm512_mask_slli_epi16(__m512i __W, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_slli_epi16
  // CHECK: @llvm.x86.avx512.pslli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_slli_epi16(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_slli_epi16_2(__m512i __W, __mmask32 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm512_mask_slli_epi16_2
  // CHECK: @llvm.x86.avx512.pslli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_slli_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_slli_epi16(__mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_slli_epi16
  // CHECK: @llvm.x86.avx512.pslli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_slli_epi16(__U, __A, 5); 
}

__m512i test_mm512_maskz_slli_epi16_2(__mmask32 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm512_maskz_slli_epi16_2
  // CHECK: @llvm.x86.avx512.pslli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_slli_epi16(__U, __A, __B); 
}

__m512i test_mm512_bslli_epi128(__m512i __A) {
  // CHECK-LABEL: @test_mm512_bslli_epi128
  // CHECK: shufflevector <64 x i8> zeroinitializer, <64 x i8> %{{.*}}, <64 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122>
  return _mm512_bslli_epi128(__A, 5);
}

__m512i test_mm512_srlv_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.512(
  return _mm512_srlv_epi16(__A, __B); 
}

__m512i test_mm512_mask_srlv_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_srlv_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srlv_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_srlv_epi16(__U, __A, __B); 
}

__m512i test_mm512_srav_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.512(
  return _mm512_srav_epi16(__A, __B); 
}

__m512i test_mm512_mask_srav_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_srav_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srav_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_srav_epi16(__U, __A, __B); 
}

__m512i test_mm512_sra_epi16(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_sra_epi16
  // CHECK: @llvm.x86.avx512.psra.w.512
  return _mm512_sra_epi16(__A, __B); 
}

__m512i test_mm512_mask_sra_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_sra_epi16
  // CHECK: @llvm.x86.avx512.psra.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_sra_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sra_epi16(__mmask32 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_sra_epi16
  // CHECK: @llvm.x86.avx512.psra.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_sra_epi16(__U, __A, __B); 
}

__m512i test_mm512_srai_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_srai_epi16
  // CHECK: @llvm.x86.avx512.psrai.w.512
  return _mm512_srai_epi16(__A, 5); 
}

__m512i test_mm512_srai_epi16_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm512_srai_epi16_2
  // CHECK: @llvm.x86.avx512.psrai.w.512
  return _mm512_srai_epi16(__A, __B); 
}

__m512i test_mm512_mask_srai_epi16(__m512i __W, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_srai_epi16
  // CHECK: @llvm.x86.avx512.psrai.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_srai_epi16(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_srai_epi16_2(__m512i __W, __mmask32 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm512_mask_srai_epi16_2
  // CHECK: @llvm.x86.avx512.psrai.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_srai_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srai_epi16(__mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_srai_epi16
  // CHECK: @llvm.x86.avx512.psrai.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_srai_epi16(__U, __A, 5); 
}

__m512i test_mm512_maskz_srai_epi16_2(__mmask32 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm512_maskz_srai_epi16_2
  // CHECK: @llvm.x86.avx512.psrai.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_srai_epi16(__U, __A, __B); 
}

__m512i test_mm512_srl_epi16(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_srl_epi16
  // CHECK: @llvm.x86.avx512.psrl.w.512
  return _mm512_srl_epi16(__A, __B); 
}

__m512i test_mm512_mask_srl_epi16(__m512i __W, __mmask32 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_srl_epi16
  // CHECK: @llvm.x86.avx512.psrl.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_srl_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srl_epi16(__mmask32 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_srl_epi16
  // CHECK: @llvm.x86.avx512.psrl.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_srl_epi16(__U, __A, __B); 
}

__m512i test_mm512_srli_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_srli_epi16
  // CHECK: @llvm.x86.avx512.psrli.w.512
  return _mm512_srli_epi16(__A, 5); 
}

__m512i test_mm512_srli_epi16_2(__m512i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm512_srli_epi16_2
  // CHECK: @llvm.x86.avx512.psrli.w.512
  return _mm512_srli_epi16(__A, __B); 
}

__m512i test_mm512_mask_srli_epi16(__m512i __W, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_srli_epi16
  // CHECK: @llvm.x86.avx512.psrli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_srli_epi16(__W, __U, __A, 5); 
}

__m512i test_mm512_mask_srli_epi16_2(__m512i __W, __mmask32 __U, __m512i __A, unsigned int __B) {
  // CHECK-LABEL: @test_mm512_mask_srli_epi16_2
  // CHECK: @llvm.x86.avx512.psrli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_srli_epi16(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srli_epi16(__mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_srli_epi16
  // CHECK: @llvm.x86.avx512.psrli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_srli_epi16(__U, __A, 5); 
}

__m512i test_mm512_maskz_srli_epi16_2(__mmask32 __U, __m512i __A, int __B) {
  // CHECK-LABEL: @test_mm512_maskz_srli_epi16_2
  // CHECK: @llvm.x86.avx512.psrli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_srli_epi16(__U, __A, __B); 
}

__m512i test_mm512_bsrli_epi128(__m512i __A) {
  // CHECK-LABEL: @test_mm512_bsrli_epi128
  // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> zeroinitializer, <64 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 82, i32 83, i32 84, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 98, i32 99, i32 100, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113, i32 114, i32 115, i32 116>
  return _mm512_bsrli_epi128(__A, 5);
}
__m512i test_mm512_mask_mov_epi16(__m512i __W, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_mov_epi16
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_mov_epi16(__W, __U, __A); 
}

__m512i test_mm512_maskz_mov_epi16(__mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_mov_epi16
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_mov_epi16(__U, __A); 
}

__m512i test_mm512_mask_mov_epi8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_mov_epi8
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_mov_epi8(__W, __U, __A); 
}

__m512i test_mm512_maskz_mov_epi8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_mov_epi8
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_mov_epi8(__U, __A); 
}

__m512i test_mm512_mask_set1_epi8(__m512i __O, __mmask64 __M, char __A) {
  // CHECK-LABEL: @test_mm512_mask_set1_epi8
  // CHECK: insertelement <64 x i8> poison, i8 %{{.*}}, i32 0
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 16
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 17
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 18
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 19
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 20
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 21
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 22
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 23
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 24
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 25
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 26
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 27
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 28
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 29
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 30
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 31
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 34
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 35
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 36
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 37
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 38
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 39
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 40
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 41
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 42
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 43
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 44
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 45
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 46
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 47
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 48
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 49
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 50
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 51
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 52
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 53
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 54
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 55
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 56
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 57
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 58
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 59
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 60
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 61
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 62
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 63
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_set1_epi8(__O, __M, __A); 
}

__m512i test_mm512_maskz_set1_epi8(__mmask64 __M, char __A) {
  // CHECK-LABEL: @test_mm512_maskz_set1_epi8
  // CHECK: insertelement <64 x i8> poison, i8 %{{.*}}, i32 0
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 16
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 17
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 18
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 19
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 20
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 21
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 22
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 23
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 24
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 25
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 26
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 27
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 28
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 29
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 30
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 31
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 32
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 33
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 34
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 35
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 36
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 37
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 38
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 39
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 40
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 41
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 42
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 43
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 44
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 45
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 46
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 47
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 48
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 49
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 50
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 51
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 52
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 53
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 54
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 55
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 56
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 57
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 58
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 59
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 60
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 61
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 62
  // CHECK: insertelement <64 x i8> %{{.*}}, i8 %{{.*}}, i32 63
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_set1_epi8(__M, __A); 
}

__mmask64 test_mm512_kunpackd(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_mm512_kunpackd
  // CHECK: [[LHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK: [[LHS2:%.*]] = shufflevector <64 x i1> [[LHS]], <64 x i1> [[LHS]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // CHECK: [[RHS2:%.*]] = shufflevector <64 x i1> [[RHS]], <64 x i1> [[RHS]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // CHECK: [[CONCAT:%.*]] = shufflevector <32 x i1> [[RHS2]], <32 x i1> [[LHS2]], <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  return _mm512_mask_cmpneq_epu8_mask(_mm512_kunpackd(_mm512_cmpneq_epu8_mask(__B, __A),_mm512_cmpneq_epu8_mask(__C, __D)), __E, __F); 
}

__mmask32 test_mm512_kunpackw(__m512i __A, __m512i __B, __m512i __C, __m512i __D, __m512i __E, __m512i __F) {
  // CHECK-LABEL: @test_mm512_kunpackw
  // CHECK: [[LHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[RHS:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: [[LHS2:%.*]] = shufflevector <32 x i1> [[LHS]], <32 x i1> [[LHS]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: [[RHS2:%.*]] = shufflevector <32 x i1> [[RHS]], <32 x i1> [[RHS]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: [[CONCAT:%.*]] = shufflevector <16 x i1> [[RHS2]], <16 x i1> [[LHS2]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  return _mm512_mask_cmpneq_epu16_mask(_mm512_kunpackw(_mm512_cmpneq_epu16_mask(__B, __A),_mm512_cmpneq_epu16_mask(__C, __D)), __E, __F); 
}

__m512i test_mm512_loadu_epi16 (void *__P)
{
  // CHECK-LABEL: @test_mm512_loadu_epi16
  // CHECK: load <8 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm512_loadu_epi16 (__P);
}

__m512i test_mm512_mask_loadu_epi16(__m512i __W, __mmask32 __U, void const *__P) {
  // CHECK-LABEL: @test_mm512_mask_loadu_epi16
  // CHECK: @llvm.masked.load.v32i16.p0(ptr %{{.*}}, i32 1, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_mask_loadu_epi16(__W, __U, __P); 
}

__m512i test_mm512_maskz_loadu_epi16(__mmask32 __U, void const *__P) {
  // CHECK-LABEL: @test_mm512_maskz_loadu_epi16
  // CHECK: @llvm.masked.load.v32i16.p0(ptr %{{.*}}, i32 1, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_maskz_loadu_epi16(__U, __P); 
}

__m512i test_mm512_loadu_epi8 (void *__P)
{
  // CHECK-LABEL: @test_mm512_loadu_epi8
  // CHECK: load <8 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm512_loadu_epi8 (__P);
}

__m512i test_mm512_mask_loadu_epi8(__m512i __W, __mmask64 __U, void const *__P) {
  // CHECK-LABEL: @test_mm512_mask_loadu_epi8
  // CHECK: @llvm.masked.load.v64i8.p0(ptr %{{.*}}, i32 1, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_mask_loadu_epi8(__W, __U, __P); 
}

__m512i test_mm512_maskz_loadu_epi8(__mmask64 __U, void const *__P) {
  // CHECK-LABEL: @test_mm512_maskz_loadu_epi8
  // CHECK: @llvm.masked.load.v64i8.p0(ptr %{{.*}}, i32 1, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_maskz_loadu_epi8(__U, __P); 
}

void test_mm512_storeu_epi16(void *__P, __m512i __A) {
  // CHECK-LABEL: @test_mm512_storeu_epi16
  // CHECK: store <8 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm512_storeu_epi16(__P, __A); 
}

void test_mm512_mask_storeu_epi16(void *__P, __mmask32 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_storeu_epi16
  // CHECK: @llvm.masked.store.v32i16.p0(<32 x i16> %{{.*}}, ptr %{{.*}}, i32 1, <32 x i1> %{{.*}})
  return _mm512_mask_storeu_epi16(__P, __U, __A);
}

__mmask64 test_mm512_test_epi8_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_test_epi8_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <64 x i8> %{{.*}}, %{{.*}}
  return _mm512_test_epi8_mask(__A, __B); 
}

void test_mm512_storeu_epi8(void *__P, __m512i __A) {
  // CHECK-LABEL: @test_mm512_storeu_epi8
  // CHECK: store <8 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm512_storeu_epi8(__P, __A);
}

void test_mm512_mask_storeu_epi8(void *__P, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_storeu_epi8
  // CHECK: @llvm.masked.store.v64i8.p0(<64 x i8> %{{.*}}, ptr %{{.*}}, i32 1, <64 x i1> %{{.*}})
  return _mm512_mask_storeu_epi8(__P, __U, __A); 
}
__mmask64 test_mm512_mask_test_epi8_mask(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_test_epi8_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_test_epi8_mask(__U, __A, __B); 
}

__mmask32 test_mm512_test_epi16_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_test_epi16_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <32 x i16> %{{.*}}, %{{.*}}
  return _mm512_test_epi16_mask(__A, __B); 
}

__mmask32 test_mm512_mask_test_epi16_mask(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_test_epi16_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_test_epi16_mask(__U, __A, __B); 
}

__mmask64 test_mm512_testn_epi8_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_testn_epi8_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  return _mm512_testn_epi8_mask(__A, __B); 
}

__mmask64 test_mm512_mask_testn_epi8_mask(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_testn_epi8_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <64 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_testn_epi8_mask(__U, __A, __B); 
}

__mmask32 test_mm512_testn_epi16_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_testn_epi16_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  return _mm512_testn_epi16_mask(__A, __B); 
}

__mmask32 test_mm512_mask_testn_epi16_mask(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_testn_epi16_mask
  // CHECK: and <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_testn_epi16_mask(__U, __A, __B); 
}

__mmask64 test_mm512_movepi8_mask(__m512i __A) {
  // CHECK-LABEL: @test_mm512_movepi8_mask
  // CHECK: [[CMP:%.*]] = icmp slt <64 x i8> %{{.*}}, zeroinitializer
  return _mm512_movepi8_mask(__A); 
}

__m512i test_mm512_movm_epi8(__mmask64 __A) {
  // CHECK-LABEL: @test_mm512_movm_epi8
  // CHECK:  %{{.*}} = bitcast i64 %{{.*}} to <64 x i1>
  // CHECK:  %vpmovm2.i = sext <64 x i1> %{{.*}} to <64 x i8>
  return _mm512_movm_epi8(__A); 
}

__m512i test_mm512_movm_epi16(__mmask32 __A) {
  // CHECK-LABEL: @test_mm512_movm_epi16
  // CHECK:  %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK:  %vpmovm2.i = sext <32 x i1> %{{.*}} to <32 x i16>
  return _mm512_movm_epi16(__A); 
}

__m512i test_mm512_broadcastb_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm512_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <64 x i32> zeroinitializer
  return _mm512_broadcastb_epi8(__A);
}

__m512i test_mm512_mask_broadcastb_epi8(__m512i __O, __mmask64 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <64 x i32> zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_broadcastb_epi8(__O, __M, __A);
}

__m512i test_mm512_maskz_broadcastb_epi8(__mmask64 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <64 x i32> zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_broadcastb_epi8(__M, __A);
}

__m512i test_mm512_broadcastw_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm512_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <32 x i32> zeroinitializer
  return _mm512_broadcastw_epi16(__A);
}

__m512i test_mm512_mask_broadcastw_epi16(__m512i __O, __mmask32 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_broadcastw_epi16(__O, __M, __A);
}

__m512i test_mm512_maskz_broadcastw_epi16(__mmask32 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_broadcastw_epi16(__M, __A);
}

__m512i test_mm512_mask_set1_epi16(__m512i __O, __mmask32 __M, short __A) {
  // CHECK-LABEL: @test_mm512_mask_set1_epi16
  // CHECK: insertelement <32 x i16> poison, i16 %{{.*}}, i32 0
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 8
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 9
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 10
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 11
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 12
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 13
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 14
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 15
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 16
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 17
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 18
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 19
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 20
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 21
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 22
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 23
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 24
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 25
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 26
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 27
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 28
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 29
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 30
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 31
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_set1_epi16(__O, __M, __A); 
}

__m512i test_mm512_maskz_set1_epi16(__mmask32 __M, short __A) {
  // CHECK-LABEL: @test_mm512_maskz_set1_epi16
  // CHECK: insertelement <32 x i16> poison, i16 %{{.*}}, i32 0
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 8
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 9
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 10
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 11
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 12
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 13
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 14
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 15
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 16
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 17
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 18
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 19
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 20
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 21
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 22
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 23
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 24
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 25
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 26
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 27
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 28
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 29
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 30
  // CHECK: insertelement <32 x i16> %{{.*}}, i16 %{{.*}}, i32 31
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_set1_epi16(__M, __A); 
}
__m512i test_mm512_permutexvar_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.512
 return _mm512_permutexvar_epi16(__A, __B); 
}

__m512i test_mm512_maskz_permutexvar_epi16(__mmask32 __M, __m512i __A, __m512i __B) {
 // CHECK-LABEL: @test_mm512_maskz_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_permutexvar_epi16(__M, __A, __B); 
}

__m512i test_mm512_mask_permutexvar_epi16(__m512i __W, __mmask32 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_permutexvar_epi16(__W, __M, __A, __B); 
}
__m512i test_mm512_alignr_epi8(__m512i __A,__m512i __B){
    // CHECK-LABEL: @test_mm512_alignr_epi8
    // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113>
    return _mm512_alignr_epi8(__A, __B, 2); 
}

__m512i test_mm512_mask_alignr_epi8(__m512i __W, __mmask64 __U, __m512i __A,__m512i __B){
    // CHECK-LABEL: @test_mm512_mask_alignr_epi8
    // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113>
    // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
    return _mm512_mask_alignr_epi8(__W, __U, __A, __B, 2); 
}

__m512i test_mm512_maskz_alignr_epi8(__mmask64 __U, __m512i __A,__m512i __B){
    // CHECK-LABEL: @test_mm512_maskz_alignr_epi8
    // CHECK: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113>
    // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
   return _mm512_maskz_alignr_epi8(__U, __A, __B, 2); 
}



__m512i test_mm512_mm_dbsad_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mm_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.512
  return _mm512_dbsad_epu8(__A, __B, 170); 
}

__m512i test_mm512_mm_mask_dbsad_epu8(__m512i __W, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mm_mask_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.512
  //CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_dbsad_epu8(__W, __U, __A, __B, 170); 
}

__m512i test_mm512_mm_maskz_dbsad_epu8(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mm_maskz_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.512
  //CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_dbsad_epu8(__U, __A, __B, 170); 
}

__m512i test_mm512_sad_epu8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_sad_epu8
  // CHECK: @llvm.x86.avx512.psad.bw.512
  return _mm512_sad_epu8(__A, __B); 
}

__mmask32 test_mm512_movepi16_mask(__m512i __A) {
  // CHECK-LABEL: @test_mm512_movepi16_mask
  // CHECK: [[CMP:%.*]] = icmp slt <32 x i16> %{{.*}}, zeroinitializer
  return _mm512_movepi16_mask(__A); 
}

void test_mm512_mask_cvtepi16_storeu_epi8 (void * __P, __mmask32 __M, __m512i __A)
{
 // CHECK-LABEL: @test_mm512_mask_cvtepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmov.wb.mem.512
 _mm512_mask_cvtepi16_storeu_epi8 ( __P,  __M, __A);
}

void test_mm512_mask_cvtsepi16_storeu_epi8 (void * __P, __mmask32 __M, __m512i __A)
{
 // CHECK-LABEL: @test_mm512_mask_cvtsepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovs.wb.mem.512
 _mm512_mask_cvtsepi16_storeu_epi8 ( __P,  __M, __A);
}

void test_mm512_mask_cvtusepi16_storeu_epi8 (void * __P, __mmask32 __M, __m512i __A)
{
 // CHECK-LABEL: @test_mm512_mask_cvtusepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovus.wb.mem.512
 _mm512_mask_cvtusepi16_storeu_epi8 ( __P, __M, __A);
}
