// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <x86intrin.h>

// _MM_PCOMCTRL_LT

__m128i test_mm_comlt_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epu8
  // CHECK: icmp ult <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comlt_epu8(a, b);
}

__m128i test_mm_comlt_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epu16
  // CHECK: icmp ult <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comlt_epu16(a, b);
}

__m128i test_mm_comlt_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epu32
  // CHECK: icmp ult <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comlt_epu32(a, b);
}

__m128i test_mm_comlt_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epu64
  // CHECK: icmp ult <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comlt_epu64(a, b);
}

__m128i test_mm_comlt_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epi8
  // CHECK: icmp slt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comlt_epi8(a, b);
}

__m128i test_mm_comlt_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epi16
  // CHECK: icmp slt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comlt_epi16(a, b);
}

__m128i test_mm_comlt_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epi32
  // CHECK: icmp slt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comlt_epi32(a, b);
}

__m128i test_mm_comlt_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comlt_epi64
  // CHECK: icmp slt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comlt_epi64(a, b);
}

// _MM_PCOMCTRL_LE

__m128i test_mm_comle_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epu8
  // CHECK: icmp ule <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comle_epu8(a, b);
}

__m128i test_mm_comle_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epu16
  // CHECK: icmp ule <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comle_epu16(a, b);
}

__m128i test_mm_comle_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epu32
  // CHECK: icmp ule <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comle_epu32(a, b);
}

__m128i test_mm_comle_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epu64
  // CHECK: icmp ule <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comle_epu64(a, b);
}

__m128i test_mm_comle_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epi8
  // CHECK: icmp sle <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comle_epi8(a, b);
}

__m128i test_mm_comle_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epi16
  // CHECK: icmp sle <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comle_epi16(a, b);
}

__m128i test_mm_comle_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epi32
  // CHECK: icmp sle <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comle_epi32(a, b);
}

__m128i test_mm_comle_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comle_epi64
  // CHECK: icmp sle <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comle_epi64(a, b);
}

// _MM_PCOMCTRL_GT

__m128i test_mm_comgt_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epu8
  // CHECK: icmp ugt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comgt_epu8(a, b);
}

__m128i test_mm_comgt_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epu16
  // CHECK: icmp ugt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comgt_epu16(a, b);
}

__m128i test_mm_comgt_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epu32
  // CHECK: icmp ugt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comgt_epu32(a, b);
}

__m128i test_mm_comgt_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epu64
  // CHECK: icmp ugt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comgt_epu64(a, b);
}

__m128i test_mm_comgt_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epi8
  // CHECK: icmp sgt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comgt_epi8(a, b);
}

__m128i test_mm_comgt_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epi16
  // CHECK: icmp sgt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comgt_epi16(a, b);
}

__m128i test_mm_comgt_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epi32
  // CHECK: icmp sgt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comgt_epi32(a, b);
}

__m128i test_mm_comgt_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comgt_epi64
  // CHECK: icmp sgt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comgt_epi64(a, b);
}

// _MM_PCOMCTRL_GE

__m128i test_mm_comge_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epu8
  // CHECK: icmp uge <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comge_epu8(a, b);
}

__m128i test_mm_comge_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epu16
  // CHECK: icmp uge <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comge_epu16(a, b);
}

__m128i test_mm_comge_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epu32
  // CHECK: icmp uge <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comge_epu32(a, b);
}

__m128i test_mm_comge_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epu64
  // CHECK: icmp uge <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comge_epu64(a, b);
}

__m128i test_mm_comge_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epi8
  // CHECK: icmp sge <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comge_epi8(a, b);
}

__m128i test_mm_comge_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epi16
  // CHECK: icmp sge <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comge_epi16(a, b);
}

__m128i test_mm_comge_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epi32
  // CHECK: icmp sge <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comge_epi32(a, b);
}

__m128i test_mm_comge_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comge_epi64
  // CHECK: icmp sge <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comge_epi64(a, b);
}

// _MM_PCOMCTRL_EQ

__m128i test_mm_comeq_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epu8
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comeq_epu8(a, b);
}

__m128i test_mm_comeq_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epu16
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comeq_epu16(a, b);
}

__m128i test_mm_comeq_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epu32
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comeq_epu32(a, b);
}

__m128i test_mm_comeq_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epu64
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comeq_epu64(a, b);
}

__m128i test_mm_comeq_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epi8
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comeq_epi8(a, b);
}

__m128i test_mm_comeq_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epi16
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comeq_epi16(a, b);
}

__m128i test_mm_comeq_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epi32
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comeq_epi32(a, b);
}

__m128i test_mm_comeq_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comeq_epi64
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comeq_epi64(a, b);
}

// _MM_PCOMCTRL_NEQ

__m128i test_mm_comneq_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epu8
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comneq_epu8(a, b);
}

__m128i test_mm_comneq_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epu16
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comneq_epu16(a, b);
}

__m128i test_mm_comneq_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epu32
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comneq_epu32(a, b);
}

__m128i test_mm_comneq_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epu64
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comneq_epu64(a, b);
}

__m128i test_mm_comneq_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epi8
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_comneq_epi8(a, b);
}

__m128i test_mm_comneq_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epi16
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_comneq_epi16(a, b);
}

__m128i test_mm_comneq_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epi32
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_comneq_epi32(a, b);
}

__m128i test_mm_comneq_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comneq_epi64
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_comneq_epi64(a, b);
}

// _MM_PCOMCTRL_FALSE

__m128i test_mm_comfalse_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epu8
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_comfalse_epu8(a, b);
}

__m128i test_mm_comfalse_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epu16
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_comfalse_epu16(a, b);
}

__m128i test_mm_comfalse_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epu32
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_comfalse_epu32(a, b);
}

__m128i test_mm_comfalse_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epu64
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_comfalse_epu64(a, b);
}

__m128i test_mm_comfalse_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epi8
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_comfalse_epi8(a, b);
}

__m128i test_mm_comfalse_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epi16
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_comfalse_epi16(a, b);
}

__m128i test_mm_comfalse_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epi32
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_comfalse_epi32(a, b);
}

__m128i test_mm_comfalse_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comfalse_epi64
  // CHECK: ret <2 x i64> zeroinitializer
  return _mm_comfalse_epi64(a, b);
}

// _MM_PCOMCTRL_TRUE

__m128i test_mm_comtrue_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epu8
  // CHECK: ret <2 x i64> splat (i64 -1)
  return _mm_comtrue_epu8(a, b);
}

__m128i test_mm_comtrue_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epu16
  // CHECK: ret <2 x i64> splat (i64 -1)
  return _mm_comtrue_epu16(a, b);
}

__m128i test_mm_comtrue_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epu32
  // CHECK: ret <2 x i64> splat (i64 -1)
  return _mm_comtrue_epu32(a, b);
}

__m128i test_mm_comtrue_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epu64
  // CHECK: ret <2 x i64> splat (i64 -1)
  return _mm_comtrue_epu64(a, b);
}

__m128i test_mm_comtrue_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epi8
  // CHECK: ret <2 x i64> splat (i64 -1)
  return _mm_comtrue_epi8(a, b);
}

__m128i test_mm_comtrue_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epi16
  // CHECK: ret <2 x i64> splat (i64 -1)
  return _mm_comtrue_epi16(a, b);
}

__m128i test_mm_comtrue_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epi32
  // CHECK: ret <2 x i64> splat (i64 -1)
  return _mm_comtrue_epi32(a, b);
}

__m128i test_mm_comtrue_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_comtrue_epi64
  // CHECK: ret <2 x i64> splat (i64 -1)
  return _mm_comtrue_epi64(a, b);
}
