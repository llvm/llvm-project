// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <x86intrin.h>

// _MM_PCOMCTRL_LT

__m128i test_mm_comlt_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 0)
  return _mm_comlt_epu8(a, b);
}

__m128i test_mm_comlt_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 0)
  return _mm_comlt_epu16(a, b);
}

__m128i test_mm_comlt_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 0)
  return _mm_comlt_epu32(a, b);
}

__m128i test_mm_comlt_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 0)
  return _mm_comlt_epu64(a, b);
}

__m128i test_mm_comlt_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 0)
  return _mm_comlt_epi8(a, b);
}

__m128i test_mm_comlt_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 0)
  return _mm_comlt_epi16(a, b);
}

__m128i test_mm_comlt_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 0)
  return _mm_comlt_epi32(a, b);
}

__m128i test_mm_comlt_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 0)
  return _mm_comlt_epi64(a, b);
}

// _MM_PCOMCTRL_LE

__m128i test_mm_comle_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 1)
  return _mm_comle_epu8(a, b);
}

__m128i test_mm_comle_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 1)
  return _mm_comle_epu16(a, b);
}

__m128i test_mm_comle_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 1)
  return _mm_comle_epu32(a, b);
}

__m128i test_mm_comle_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 1)
  return _mm_comle_epu64(a, b);
}

__m128i test_mm_comle_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 1)
  return _mm_comle_epi8(a, b);
}

__m128i test_mm_comle_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 1)
  return _mm_comle_epi16(a, b);
}

__m128i test_mm_comle_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 1)
  return _mm_comle_epi32(a, b);
}

__m128i test_mm_comle_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 1)
  return _mm_comle_epi64(a, b);
}

// _MM_PCOMCTRL_GT

__m128i test_mm_comgt_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 2)
  return _mm_comgt_epu8(a, b);
}

__m128i test_mm_comgt_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 2)
  return _mm_comgt_epu16(a, b);
}

__m128i test_mm_comgt_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_comgt_epu32(a, b);
}

__m128i test_mm_comgt_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_comgt_epu64(a, b);
}

__m128i test_mm_comgt_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 2)
  return _mm_comgt_epi8(a, b);
}

__m128i test_mm_comgt_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 2)
  return _mm_comgt_epi16(a, b);
}

__m128i test_mm_comgt_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 2)
  return _mm_comgt_epi32(a, b);
}

__m128i test_mm_comgt_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 2)
  return _mm_comgt_epi64(a, b);
}

// _MM_PCOMCTRL_GE

__m128i test_mm_comge_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 3)
  return _mm_comge_epu8(a, b);
}

__m128i test_mm_comge_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 3)
  return _mm_comge_epu16(a, b);
}

__m128i test_mm_comge_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 3)
  return _mm_comge_epu32(a, b);
}

__m128i test_mm_comge_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 3)
  return _mm_comge_epu64(a, b);
}

__m128i test_mm_comge_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 3)
  return _mm_comge_epi8(a, b);
}

__m128i test_mm_comge_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 3)
  return _mm_comge_epi16(a, b);
}

__m128i test_mm_comge_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 3)
  return _mm_comge_epi32(a, b);
}

__m128i test_mm_comge_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 3)
  return _mm_comge_epi64(a, b);
}

// _MM_PCOMCTRL_EQ

__m128i test_mm_comeq_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 4)
  return _mm_comeq_epu8(a, b);
}

__m128i test_mm_comeq_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 4)
  return _mm_comeq_epu16(a, b);
}

__m128i test_mm_comeq_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 4)
  return _mm_comeq_epu32(a, b);
}

__m128i test_mm_comeq_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 4)
  return _mm_comeq_epu64(a, b);
}

__m128i test_mm_comeq_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 4)
  return _mm_comeq_epi8(a, b);
}

__m128i test_mm_comeq_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 4)
  return _mm_comeq_epi16(a, b);
}

__m128i test_mm_comeq_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 4)
  return _mm_comeq_epi32(a, b);
}

__m128i test_mm_comeq_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 4)
  return _mm_comeq_epi64(a, b);
}

// _MM_PCOMCTRL_NEQ

__m128i test_mm_comneq_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 5)
  return _mm_comneq_epu8(a, b);
}

__m128i test_mm_comneq_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 5)
  return _mm_comneq_epu16(a, b);
}

__m128i test_mm_comneq_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 5)
  return _mm_comneq_epu32(a, b);
}

__m128i test_mm_comneq_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 5)
  return _mm_comneq_epu64(a, b);
}

__m128i test_mm_comneq_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 5)
  return _mm_comneq_epi8(a, b);
}

__m128i test_mm_comneq_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 5)
  return _mm_comneq_epi16(a, b);
}

__m128i test_mm_comneq_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 5)
  return _mm_comneq_epi32(a, b);
}

__m128i test_mm_comneq_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 5)
  return _mm_comneq_epi64(a, b);
}

// _MM_PCOMCTRL_FALSE

__m128i test_mm_comfalse_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 6)
  return _mm_comfalse_epu8(a, b);
}

__m128i test_mm_comfalse_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 6)
  return _mm_comfalse_epu16(a, b);
}

__m128i test_mm_comfalse_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 6)
  return _mm_comfalse_epu32(a, b);
}

__m128i test_mm_comfalse_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 6)
  return _mm_comfalse_epu64(a, b);
}

__m128i test_mm_comfalse_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 6)
  return _mm_comfalse_epi8(a, b);
}

__m128i test_mm_comfalse_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 6)
  return _mm_comfalse_epi16(a, b);
}

__m128i test_mm_comfalse_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 6)
  return _mm_comfalse_epi32(a, b);
}

__m128i test_mm_comfalse_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 6)
  return _mm_comfalse_epi64(a, b);
}

// _MM_PCOMCTRL_TRUE

__m128i test_mm_comtrue_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_comtrue_epu8(a, b);
}

__m128i test_mm_comtrue_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 7)
  return _mm_comtrue_epu16(a, b);
}

__m128i test_mm_comtrue_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 7)
  return _mm_comtrue_epu32(a, b);
}

__m128i test_mm_comtrue_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 7)
  return _mm_comtrue_epu64(a, b);
}

__m128i test_mm_comtrue_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_comtrue_epi8(a, b);
}

__m128i test_mm_comtrue_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 7)
  return _mm_comtrue_epi16(a, b);
}

__m128i test_mm_comtrue_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 7)
  return _mm_comtrue_epi32(a, b);
}

__m128i test_mm_comtrue_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 7)
  return _mm_comtrue_epi64(a, b);
}
