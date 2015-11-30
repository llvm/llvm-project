// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse2 -emit-llvm -o - -Werror | FileCheck %s --check-prefix=DAG
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse2 -fno-signed-char -emit-llvm -o - -Werror | FileCheck %s --check-prefix=DAG
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse2 -S -o - -Werror | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse2 -fno-signed-char -S -o - -Werror | FileCheck %s --check-prefix=ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128i test_mm_add_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_add_epi8
  // DAG: add <16 x i8>
  //
  // ASM-LABEL: test_mm_add_epi8
  // ASM: paddb
  return _mm_add_epi8(A, B);
}

__m128i test_mm_add_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_add_epi16
  // DAG: add <8 x i16>
  //
  // ASM-LABEL: test_mm_add_epi16
  // ASM: paddw
  return _mm_add_epi16(A, B);
}

__m128i test_mm_add_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_add_epi32
  // DAG: add <4 x i32>
  //
  // ASM-LABEL: test_mm_add_epi32
  // ASM: paddd
  return _mm_add_epi32(A, B);
}

__m128i test_mm_add_epi64(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_add_epi64
  // DAG: add <2 x i64>
  //
  // ASM-LABEL: test_mm_add_epi64
  // ASM: paddq
  return _mm_add_epi64(A, B);
}

__m128d test_mm_add_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_add_pd
  // DAG: fadd <2 x double>
  //
  // ASM-LABEL: test_mm_add_pd
  // ASM: addpd
  return _mm_add_pd(A, B);
}

__m128d test_mm_add_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_add_sd
  // DAG: fadd double
  //
  // ASM-LABEL: test_mm_add_sd
  // ASM: addsd
  return _mm_add_sd(A, B);
}

__m128i test_mm_adds_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_adds_epi8
  // DAG: call <16 x i8> @llvm.x86.sse2.padds.b
  //
  // ASM-LABEL: test_mm_adds_epi8
  // ASM: paddsb
  return _mm_adds_epi8(A, B);
}

__m128i test_mm_adds_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_adds_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.padds.w
  //
  // ASM-LABEL: test_mm_adds_epi16
  // ASM: paddsw
  return _mm_adds_epi16(A, B);
}

__m128i test_mm_adds_epu8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_adds_epu8
  // DAG: call <16 x i8> @llvm.x86.sse2.paddus.b
  //
  // ASM-LABEL: test_mm_adds_epu8
  // ASM: paddusb
  return _mm_adds_epu8(A, B);
}

__m128i test_mm_adds_epu16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_adds_epu16
  // DAG: call <8 x i16> @llvm.x86.sse2.paddus.w
  //
  // ASM-LABEL: test_mm_adds_epu16
  // ASM: paddusw
  return _mm_adds_epu16(A, B);
}

__m128d test_mm_and_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_and_pd
  // DAG: and <4 x i32>
  //
  // ASM-LABEL: test_mm_and_pd
  // ASM: pand
  return _mm_and_pd(A, B);
}

__m128i test_mm_and_si128(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_and_si128
  // DAG: and <2 x i64>
  //
  // ASM-LABEL: test_mm_and_si128
  // ASM: andps
  return _mm_and_si128(A, B);
}

__m128i test_mm_avg_epu8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_avg_epu8
  // DAG: call <16 x i8> @llvm.x86.sse2.pavg.b
  //
  // ASM-LABEL: test_mm_avg_epu8
  // ASM: pavgb
  return _mm_avg_epu8(A, B);
}

__m128i test_mm_avg_epu16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_avg_epu16
  // DAG: call <8 x i16> @llvm.x86.sse2.pavg.w
  //
  // ASM-LABEL: test_mm_avg_epu16
  // ASM: pavgw
  return _mm_avg_epu16(A, B);
}

__m128i test_mm_bslli_si128(__m128i A) {
  // DAG-LABEL: test_mm_bslli_si128
  // DAG: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  //
  // ASM-LABEL: test_mm_bslli_si128
  // ASM: pslldq $5, %xmm{{.*}}
  return _mm_bslli_si128(A, 5);
}

__m128i test_mm_bsrli_si128(__m128i A) {
  // DAG-LABEL: test_mm_bsrli_si128
  // DAG: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  //
  // ASM-LABEL: test_mm_bsrli_si128
  // ASM: psrldq $5, %xmm{{.*}}
  return _mm_bsrli_si128(A, 5);
}

void test_mm_clflush(void* A) {
  // DAG-LABEL: test_mm_clflush
  // DAG: call void @llvm.x86.sse2.clflush(i8* %{{.*}})
  //
  // ASM-LABEL: test_mm_clflush
  // ASM: clflush
  _mm_clflush(A);
}

__m128i test_mm_cmpeq_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmpeq_epi8
  // DAG: icmp eq <16 x i8>
  //
  // ASM-LABEL: test_mm_cmpeq_epi8
  // ASM: pcmpeqb
  return _mm_cmpeq_epi8(A, B);
}

__m128i test_mm_cmpeq_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmpeq_epi16
  // DAG: icmp eq <8 x i16>
  //
  // ASM-LABEL: test_mm_cmpeq_epi16
  // ASM: pcmpeqw
  return _mm_cmpeq_epi16(A, B);
}

__m128i test_mm_cmpeq_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmpeq_epi32
  // DAG: icmp eq <4 x i32>
  //
  // ASM-LABEL: test_mm_cmpeq_epi32
  // ASM: pcmpeqd
  return _mm_cmpeq_epi32(A, B);
}

__m128d test_mm_cmpeq_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpeq_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 0)
  //
  // ASM-LABEL: test_mm_cmpeq_pd
  // ASM: cmpeqpd
  return _mm_cmpeq_pd(A, B);
}

__m128d test_mm_cmpeq_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpeq_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 0)
  //
  // ASM-LABEL: test_mm_cmpeq_sd
  // ASM: cmpeqsd
  return _mm_cmpeq_sd(A, B);
}

__m128d test_mm_cmpge_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpge_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  //
  // ASM-LABEL: test_mm_cmpge_pd
  // ASM: cmplepd
  return _mm_cmpge_pd(A, B);
}

__m128d test_mm_cmpge_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpge_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  //
  // ASM-LABEL: test_mm_cmpge_sd
  // ASM: cmplesd
  return _mm_cmpge_sd(A, B);
}

__m128i test_mm_cmpgt_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmpgt_epi8
  // DAG: icmp sgt <16 x i8>
  //
  // ASM-LABEL: test_mm_cmpgt_epi8
  // ASM: pcmpgtb
  return _mm_cmpgt_epi8(A, B);
}

__m128i test_mm_cmpgt_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmpgt_epi16
  // DAG: icmp sgt <8 x i16>
  //
  // ASM-LABEL: test_mm_cmpgt_epi16
  // ASM: pcmpgtw
  return _mm_cmpgt_epi16(A, B);
}

__m128i test_mm_cmpgt_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmpgt_epi32
  // DAG: icmp sgt <4 x i32>
  //
  // ASM-LABEL: test_mm_cmpgt_epi32
  // ASM: pcmpgtd
  return _mm_cmpgt_epi32(A, B);
}

__m128d test_mm_cmpgt_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpgt_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  //
  // ASM-LABEL: test_mm_cmpgt_pd
  // ASM: cmpltpd
  return _mm_cmpgt_pd(A, B);
}

__m128d test_mm_cmpgt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpgt_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  //
  // ASM-LABEL: test_mm_cmpgt_sd
  // ASM: cmpltsd
  return _mm_cmpgt_sd(A, B);
}

__m128d test_mm_cmple_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmple_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  //
  // ASM-LABEL: test_mm_cmple_pd
  // ASM: cmplepd
  return _mm_cmple_pd(A, B);
}

__m128d test_mm_cmple_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmple_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  //
  // ASM-LABEL: test_mm_cmple_sd
  // ASM: cmplesd
  return _mm_cmple_sd(A, B);
}

__m128i test_mm_cmplt_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmplt_epi8
  // DAG: icmp sgt <16 x i8>
  //
  // ASM-LABEL: test_mm_cmplt_epi8
  // ASM: pcmpgtb
  return _mm_cmplt_epi8(A, B);
}

__m128i test_mm_cmplt_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmplt_epi16
  // DAG: icmp sgt <8 x i16>
  //
  // ASM-LABEL: test_mm_cmplt_epi16
  // ASM: pcmpgtw
  return _mm_cmplt_epi16(A, B);
}

__m128i test_mm_cmplt_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_cmplt_epi32
  // DAG: icmp sgt <4 x i32>
  //
  // ASM-LABEL: test_mm_cmplt_epi32
  // ASM: pcmpgtd
  return _mm_cmplt_epi32(A, B);
}

__m128d test_mm_cmplt_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmplt_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  //
  // ASM-LABEL: test_mm_cmplt_pd
  // ASM: cmpltpd
  return _mm_cmplt_pd(A, B);
}

__m128d test_mm_cmplt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmplt_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  //
  // ASM-LABEL: test_mm_cmplt_sd
  // ASM: cmpltsd
  return _mm_cmplt_sd(A, B);
}

__m128d test_mm_cmpneq_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpneq_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 4)
  //
  // ASM-LABEL: test_mm_cmpneq_pd
  // ASM: cmpneqpd
  return _mm_cmpneq_pd(A, B);
}

__m128d test_mm_cmpneq_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpneq_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 4)
  //
  // ASM-LABEL: test_mm_cmpneq_sd
  // ASM: cmpneqsd
  return _mm_cmpneq_sd(A, B);
}

__m128d test_mm_cmpnge_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpnge_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  //
  // ASM-LABEL: test_mm_cmpnge_pd
  // ASM: cmpnlepd
  return _mm_cmpnge_pd(A, B);
}

__m128d test_mm_cmpnge_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpnge_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  //
  // ASM-LABEL: test_mm_cmpnge_sd
  // ASM: cmpnlesd
  return _mm_cmpnge_sd(A, B);
}

__m128d test_mm_cmpngt_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpngt_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  //
  // ASM-LABEL: test_mm_cmpngt_pd
  // ASM: cmpnltpd
  return _mm_cmpngt_pd(A, B);
}

__m128d test_mm_cmpngt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpngt_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  //
  // ASM-LABEL: test_mm_cmpngt_sd
  // ASM: cmpnltsd
  return _mm_cmpngt_sd(A, B);
}

__m128d test_mm_cmpnle_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpnle_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  //
  // ASM-LABEL: test_mm_cmpnle_pd
  // ASM: cmpnlepd
  return _mm_cmpnle_pd(A, B);
}

__m128d test_mm_cmpnle_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpnle_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  //
  // ASM-LABEL: test_mm_cmpnle_sd
  // ASM: cmpnlesd
  return _mm_cmpnle_sd(A, B);
}

__m128d test_mm_cmpnlt_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpnlt_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  //
  // ASM-LABEL: test_mm_cmpnlt_pd
  // ASM: cmpnltpd
  return _mm_cmpnlt_pd(A, B);
}

__m128d test_mm_cmpnlt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpnlt_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  //
  // ASM-LABEL: test_mm_cmpnlt_sd
  // ASM: cmpnltsd
  return _mm_cmpnlt_sd(A, B);
}

__m128d test_mm_cmpord_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpord_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 7)
  //
  // ASM-LABEL: test_mm_cmpord_pd
  // ASM: cmpordpd
  return _mm_cmpord_pd(A, B);
}

__m128d test_mm_cmpord_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpord_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 7)
  //
  // ASM-LABEL: test_mm_cmpord_sd
  // ASM: cmpordsd
  return _mm_cmpord_sd(A, B);
}

__m128d test_mm_cmpunord_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpunord_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 3)
  //
  // ASM-LABEL: test_mm_cmpunord_pd
  // ASM: cmpunordpd
  return _mm_cmpunord_pd(A, B);
}

__m128d test_mm_cmpunord_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_cmpunord_sd
  // DAG: call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 3)
  //
  // ASM-LABEL: test_mm_cmpunord_sd
  // ASM: cmpunordsd
  return _mm_cmpunord_sd(A, B);
}

int test_mm_comieq_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_comieq_sd
  // DAG: call i32 @llvm.x86.sse2.comieq.sd
  //
  // ASM-LABEL: test_mm_comieq_sd
  // ASM: comisd
  return _mm_comieq_sd(A, B);
}

int test_mm_comige_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_comige_sd
  // DAG: call i32 @llvm.x86.sse2.comige.sd
  //
  // ASM-LABEL: test_mm_comige_sd
  // ASM: comisd
  return _mm_comige_sd(A, B);
}

int test_mm_comigt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_comigt_sd
  // DAG: call i32 @llvm.x86.sse2.comigt.sd
  //
  // ASM-LABEL: test_mm_comigt_sd
  // ASM: comisd
  return _mm_comigt_sd(A, B);
}

int test_mm_comile_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_comile_sd
  // DAG: call i32 @llvm.x86.sse2.comile.sd
  //
  // ASM-LABEL: test_mm_comile_sd
  // ASM: comisd
  return _mm_comile_sd(A, B);
}

int test_mm_comilt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_comilt_sd
  // DAG: call i32 @llvm.x86.sse2.comilt.sd
  //
  // ASM-LABEL: test_mm_comilt_sd
  // ASM: comisd
  return _mm_comilt_sd(A, B);
}

int test_mm_comineq_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_comineq_sd
  // DAG: call i32 @llvm.x86.sse2.comineq.sd
  //
  // ASM-LABEL: test_mm_comineq_sd
  // ASM: comisd
  return _mm_comineq_sd(A, B);
}

__m128d test_mm_cvtepi32_pd(__m128i A) {
  // DAG-LABEL: test_mm_cvtepi32_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cvtdq2pd
  //
  // ASM-LABEL: test_mm_cvtepi32_pd
  // ASM: cvtdq2pd
  return _mm_cvtepi32_pd(A);
}

__m128 test_mm_cvtepi32_ps(__m128i A) {
  // DAG-LABEL: test_mm_cvtepi32_ps
  // DAG: call <4 x float> @llvm.x86.sse2.cvtdq2ps
  //
  // ASM-LABEL: test_mm_cvtepi32_ps
  // ASM: cvtdq2ps
  return _mm_cvtepi32_ps(A);
}

__m128i test_mm_cvtpd_epi32(__m128d A) {
  // DAG-LABEL: test_mm_cvtpd_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.cvtpd2dq
  //
  // ASM-LABEL: test_mm_cvtpd_epi32
  // ASM: cvtpd2dq
  return _mm_cvtpd_epi32(A);
}

__m128 test_mm_cvtpd_ps(__m128d A) {
  // DAG-LABEL: test_mm_cvtpd_ps
  // DAG: call <4 x float> @llvm.x86.sse2.cvtpd2ps
  //
  // ASM-LABEL: test_mm_cvtpd_ps
  // ASM: cvtpd2ps
  return _mm_cvtpd_ps(A);
}

__m128i test_mm_cvtps_epi32(__m128 A) {
  // DAG-LABEL: test_mm_cvtps_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.cvtps2dq
  //
  // ASM-LABEL: test_mm_cvtps_epi32
  // ASM: cvtps2dq
  return _mm_cvtps_epi32(A);
}

__m128d test_mm_cvtps_pd(__m128 A) {
  // DAG-LABEL: test_mm_cvtps_pd
  // DAG: call <2 x double> @llvm.x86.sse2.cvtps2pd
  //
  // ASM-LABEL: test_mm_cvtps_pd
  // ASM: cvtps2pd
  return _mm_cvtps_pd(A);
}

double test_mm_cvtsd_f64(__m128d A) {
  // DAG-LABEL: test_mm_cvtsd_f64
  // DAG: extractelement <2 x double> %{{.*}}, i32 0
  //
  // ASM-LABEL: test_mm_cvtsd_f64
  // ASM: movapd
  return _mm_cvtsd_f64(A);
}

int test_mm_cvtsd_si32(__m128d A) {
  // DAG-LABEL: test_mm_cvtsd_si32
  // DAG: call i32 @llvm.x86.sse2.cvtsd2si
  //
  // ASM-LABEL: test_mm_cvtsd_si32
  // ASM: cvtsd2si
  return _mm_cvtsd_si32(A);
}

long long test_mm_cvtsd_si64(__m128d A) {
  // DAG-LABEL: test_mm_cvtsd_si64
  // DAG: call i64 @llvm.x86.sse2.cvtsd2si64
  //
  // ASM-LABEL: test_mm_cvtsd_si64
  // ASM: cvtsd2si
  return _mm_cvtsd_si64(A);
}

__m128 test_mm_cvtsd_ss(__m128 A, __m128d B) {
  // DAG-LABEL: test_mm_cvtsd_ss
  // DAG: fptrunc double %{{.*}} to float
  //
  // ASM-LABEL: test_mm_cvtsd_ss
  // ASM: cvtsd2ss
  return _mm_cvtsd_ss(A, B);
}

int test_mm_cvtsi128_si32(__m128i A) {
  // DAG-LABEL: test_mm_cvtsi128_si32
  // DAG: extractelement <4 x i32> %{{.*}}, i32 0
  //
  // ASM-LABEL: test_mm_cvtsi128_si32
  // ASM: movd
  return _mm_cvtsi128_si32(A);
}

long long test_mm_cvtsi128_si64(__m128i A) {
  // DAG-LABEL: test_mm_cvtsi128_si64
  // DAG: extractelement <2 x i64> %{{.*}}, i32 0
  //
  // ASM-LABEL: test_mm_cvtsi128_si64
  // ASM: movd
  return _mm_cvtsi128_si64(A);
}

__m128d test_mm_cvtsi32_sd(__m128d A, int B) {
  // DAG-LABEL: test_mm_cvtsi32_sd
  // DAG: sitofp i32 %{{.*}} to double
  // DAG: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  //
  // ASM-LABEL: test_mm_cvtsi32_sd
  // ASM: cvtsi2sdl
  return _mm_cvtsi32_sd(A, B);
}

__m128i test_mm_cvtsi32_si128(int A) {
  // DAG-LABEL: test_mm_cvtsi32_si128
  // DAG: insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
  //
  // ASM-LABEL: test_mm_cvtsi32_si128
  // ASM: movd
  return _mm_cvtsi32_si128(A);
}

__m128d test_mm_cvtsi64_sd(__m128d A, long long B) {
  // DAG-LABEL: test_mm_cvtsi64_sd
  // DAG: sitofp i64 %{{.*}} to double
  // DAG: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  //
  // ASM-LABEL: test_mm_cvtsi64_sd
  // ASM: cvtsi2sdq
  return _mm_cvtsi64_sd(A, B);
}

__m128i test_mm_cvtsi64_si128(long long A) {
  // DAG-LABEL: test_mm_cvtsi64_si128
  // DAG: insertelement <2 x i64> undef, i64 %{{.*}}, i32 0
  //
  // ASM-LABEL: test_mm_cvtsi64_si128
  // ASM: movq
  return _mm_cvtsi64_si128(A);
}

__m128d test_mm_cvtss_sd(__m128d A, __m128 B) {
  // DAG-LABEL: test_mm_cvtss_sd
  // DAG: extractelement <4 x float> %{{.*}}, i32 0
  // DAG: fpext float %{{.*}} to double
  // DAG: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 0
  //
  // ASM-LABEL: test_mm_cvtss_sd
  // ASM: cvtss2sd
  return _mm_cvtss_sd(A, B);
}

__m128i test_mm_cvttpd_epi32(__m128d A) {
  // DAG-LABEL: test_mm_cvttpd_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.cvttpd2dq
  //
  // ASM-LABEL: test_mm_cvttpd_epi32
  // ASM: cvttpd2dq
  return _mm_cvttpd_epi32(A);
}

__m128i test_mm_cvttps_epi32(__m128 A) {
  // DAG-LABEL: test_mm_cvttps_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.cvttps2dq
  //
  // ASM-LABEL: test_mm_cvttps_epi32
  // ASM: cvttps2dq
  return _mm_cvttps_epi32(A);
}

int test_mm_cvttsd_si32(__m128d A) {
  // DAG-LABEL: test_mm_cvttsd_si32
  // DAG: extractelement <2 x double> %{{.*}}, i32 0
  // DAG: fptosi double %{{.*}} to i32
  //
  // ASM-LABEL: test_mm_cvttsd_si32
  // ASM: cvttsd2si
  return _mm_cvttsd_si32(A);
}

long long test_mm_cvttsd_si64(__m128d A) {
  // DAG-LABEL: test_mm_cvttsd_si64
  // DAG: extractelement <2 x double> %{{.*}}, i32 0
  // DAG: fptosi double %{{.*}} to i64
  //
  // ASM-LABEL: test_mm_cvttsd_si64
  // ASM: cvttsd2si
  return _mm_cvttsd_si64(A);
}

__m128d test_mm_div_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_div_pd
  // DAG: fdiv <2 x double>
  //
  // ASM-LABEL: test_mm_div_pd
  // ASM: divpd
  return _mm_div_pd(A, B);
}

__m128d test_mm_div_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_div_sd
  // DAG: fdiv double
  //
  // ASM-LABEL: test_mm_div_sd
  // ASM: divsd
  return _mm_div_sd(A, B);
}

// Lowering to pextrw requires optimization.
int test_mm_extract_epi16(__m128i A) {
  // DAG-LABEL: test_mm_extract_epi16
  // DAG: [[x:%.*]] = and i32 %{{.*}}, 7
  // DAG: extractelement <8 x i16> %{{.*}}, i32 [[x]]
  //
  // ASM-LABEL: test_mm_extract_epi16
  // ASM: movzwl
  return _mm_extract_epi16(A, 8);
}

// Lowering to pinsrw requires optimization.
__m128i test_mm_insert_epi16(__m128i A, short B) {
  // DAG-LABEL: test_mm_insert_epi16
  // DAG: [[x:%.*]] = and i32 %{{.*}}, 7
  // DAG: insertelement <8 x i16> %{{.*}}, i32 [[x]]
  //
  // ASM-LABEL: test_mm_insert_epi16
  // ASM: movw
  return _mm_insert_epi16(A, B, 8);
}

void test_mm_lfence() {
  // DAG-LABEL: test_mm_lfence
  // DAG: call void @llvm.x86.sse2.lfence()
  //
  // ASM-LABEL: test_mm_lfence
  // ASM: lfence
  _mm_lfence();
}

__m128d test_mm_load_pd(double const* A) {
  // DAG-LABEL: test_mm_load_pd
  // DAG: load <2 x double>, <2 x double>* %{{.*}}, align 16
  //
  // ASM-LABEL: test_mm_load_pd
  // ASM: movapd
  return _mm_load_pd(A);
}

__m128d test_mm_load_sd(double const* A) {
  // DAG-LABEL: test_mm_load_sd
  // DAG: load double, double* %{{.*}}, align 1
  //
  // ASM-LABEL: test_mm_load_sd
  // ASM: movsd
  return _mm_load_sd(A);
}

__m128i test_mm_load_si128(__m128i const* A) {
  // DAG-LABEL: test_mm_load_si128
  // DAG: load <2 x i64>, <2 x i64>* %{{.*}}, align 16
  //
  // ASM-LABEL: test_mm_load_si128
  // ASM: movdqa
  return _mm_load_si128(A);
}

__m128d test_mm_load1_pd(double const* A) {
  // DAG-LABEL: test_mm_load1_pd
  // DAG: load double, double* %{{.*}}, align 8
  // DAG: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // DAG: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  //
  // ASM-LABEL: test_mm_load1_pd
  // ASM: movsd
  // ASM: movlhps
  return _mm_load1_pd(A);
}

__m128d test_mm_loadh_pd(__m128d x, void* y) {
  // DAG-LABEL: test_mm_loadh_pd
  // DAG: load double, double* %{{.*}}, align 1{{$}}
  //
  // ASM-LABEL: test_mm_loadh_pd
  // ASM: movsd
  // ASM: unpcklpd
  return _mm_loadh_pd(x, y);
}

__m128d test_mm_loadr_pd(double const* A) {
  // DAG-LABEL: test_mm_loadr_pd
  // DAG: load <2 x double>, <2 x double>* %{{.*}}, align 16
  // DAG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 0>
  //
  // ASM-LABEL: test_mm_loadr_pd
  // ASM: movapd
  // ASM: shufpd $1,
  return _mm_loadr_pd(A);
}

__m128d test_mm_loadu_pd(double const* A) {
  // DAG-LABEL: test_mm_loadu_pd
  // DAG: load <2 x double>, <2 x double>* %{{.*}}, align 1
  //
  // ASM-LABEL: test_mm_loadu_pd
  // ASM: movupd
  return _mm_loadu_pd(A);
}

__m128i test_mm_loadu_si128(__m128i const* A) {
  // DAG-LABEL: test_mm_loadu_si128
  // DAG: load <2 x i64>, <2 x i64>* %{{.*}}, align 1
  //
  // ASM-LABEL: test_mm_loadu_si128
  // ASM: movdqu
  return _mm_loadu_si128(A);
}

__m128i test_mm_madd_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_madd_epi16
  // DAG: call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  //
  // ASM-LABEL: test_mm_madd_epi16
  // ASM: pmaddwd
  return _mm_madd_epi16(A, B);
}

void test_mm_maskmoveu_si128(__m128i A, __m128i B, char* C) {
  // DAG-LABEL: test_mm_maskmoveu_si128
  // DAG: call void @llvm.x86.sse2.maskmov.dqu(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8* %{{.*}})
  //
  // ASM-LABEL: test_mm_maskmoveu_si128
  // ASM: maskmovdqu
  _mm_maskmoveu_si128(A, B, C);
}

__m128i test_mm_max_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_max_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.pmaxs.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  //
  // ASM-LABEL: test_mm_max_epi16
  // ASM: pmaxsw
  return _mm_max_epi16(A, B);
}

__m128i test_mm_max_epu8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_max_epu8
  // DAG: call <16 x i8> @llvm.x86.sse2.pmaxu.b(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  //
  // ASM-LABEL: test_mm_max_epu8
  // ASM: pmaxub
  return _mm_max_epu8(A, B);
}

__m128d test_mm_max_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_max_pd
  // DAG: call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  //
  // ASM-LABEL: test_mm_max_pd
  // ASM: maxpd
  return _mm_max_pd(A, B);
}

__m128d test_mm_max_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_max_sd
  // DAG: call <2 x double> @llvm.x86.sse2.max.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  //
  // ASM-LABEL: test_mm_max_sd
  // ASM: maxsd
  return _mm_max_sd(A, B);
}

void test_mm_mfence() {
  // DAG-LABEL: test_mm_mfence
  // DAG: call void @llvm.x86.sse2.mfence()
  //
  // ASM-LABEL: test_mm_mfence
  // ASM: mfence
  _mm_mfence();
}

__m128i test_mm_min_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_min_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  //
  // ASM-LABEL: test_mm_min_epi16
  // ASM: pminsw
  return _mm_min_epi16(A, B);
}

__m128i test_mm_min_epu8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_min_epu8
  // DAG: call <16 x i8> @llvm.x86.sse2.pminu.b(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  //
  // ASM-LABEL: test_mm_min_epu8
  // ASM: pminub
  return _mm_min_epu8(A, B);
}

__m128d test_mm_min_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_min_pd
  // DAG: call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  //
  // ASM-LABEL: test_mm_min_pd
  // ASM: minpd
  return _mm_min_pd(A, B);
}

__m128d test_mm_min_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_min_sd
  // DAG: call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  //
  // ASM-LABEL: test_mm_min_sd
  // ASM: minsd
  return _mm_min_sd(A, B);
}

int test_mm_movemask_epi8(__m128i A) {
  // DAG-LABEL: test_mm_movemask_epi8
  // DAG: call i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8> %{{.*}})
  //
  // ASM-LABEL: test_mm_movemask_epi8
  // ASM: pmovmskb
  return _mm_movemask_epi8(A);
}

int test_mm_movemask_pd(__m128d A) {
  // DAG-LABEL: test_mm_movemask_pd
  // DAG: call i32 @llvm.x86.sse2.movmsk.pd(<2 x double> %{{.*}})
  //
  // ASM-LABEL: test_mm_movemask_pd
  // ASM: movmskpd
  return _mm_movemask_pd(A);
}

__m128i test_mm_mul_epu32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_mul_epu32
  // DAG: call <2 x i64> @llvm.x86.sse2.pmulu.dq(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  //
  // ASM-LABEL: test_mm_mul_epu32
  // ASM: pmuludq
  return _mm_mul_epu32(A, B);
}

__m128d test_mm_mul_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_mul_pd
  // DAG: fmul <2 x double> %{{.*}}, %{{.*}}
  //
  // ASM-LABEL: test_mm_mul_pd
  // ASM: mulpd
  return _mm_mul_pd(A, B);
}

__m128d test_mm_mul_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_mul_sd
  // DAG: fmul double %{{.*}}, %{{.*}}
  //
  // ASM-LABEL: test_mm_mul_sd
  // ASM: mulsd
  return _mm_mul_sd(A, B);
}

__m128i test_mm_mulhi_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_mulhi_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  //
  // ASM-LABEL: test_mm_mulhi_epi16
  // ASM: pmulhw
  return _mm_mulhi_epi16(A, B);
}

__m128i test_mm_mulhi_epu16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_mulhi_epu16
  // DAG: call <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  //
  // ASM-LABEL: test_mm_mulhi_epu16
  // ASM: pmulhuw
  return _mm_mulhi_epu16(A, B);
}

__m128i test_mm_mullo_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_mullo_epi16
  // DAG: mul <8 x i16> %{{.*}}, %{{.*}}
  //
  // ASM-LABEL: test_mm_mullo_epi16
  // ASM: pmullw
  return _mm_mullo_epi16(A, B);
}

__m128d test_mm_or_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_or_pd
  // DAG: or <4 x i32> %{{.*}}, %{{.*}}
  //
  // ASM-LABEL: test_mm_or_pd
  // ASM: por
  return _mm_or_pd(A, B);
}

__m128i test_mm_or_si128(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_or_si128
  // DAG: or <2 x i64> %{{.*}}, %{{.*}}
  //
  // ASM-LABEL: test_mm_or_si128
  // ASM: orps
  return _mm_or_si128(A, B);
}

__m128i test_mm_packs_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_packs_epi16
  // DAG: call <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  //
  // ASM-LABEL: test_mm_packs_epi16
  // ASM: packsswb
  return _mm_packs_epi16(A, B);
}

__m128i test_mm_packs_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_packs_epi32
  // DAG: call <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  //
  // ASM-LABEL: test_mm_packs_epi32
  // ASM: packssdw
  return _mm_packs_epi32(A, B);
}

__m128i test_mm_packus_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_packus_epi16
  // DAG: call <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  //
  // ASM-LABEL: test_mm_packus_epi16
  // ASM: packuswb
  return _mm_packus_epi16(A, B);
}

void test_mm_pause() {
  // DAG-LABEL: test_mm_pause
  // DAG: call void @llvm.x86.sse2.pause()
  //
  // ASM-LABEL: test_mm_pause
  // ASM: pause
  return _mm_pause();
}

__m128i test_mm_sad_epu8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sad_epu8
  // DAG: call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  //
  // ASM-LABEL: test_mm_sad_epu8
  // ASM: psadbw
  return _mm_sad_epu8(A, B);
}

__m128d test_mm_setzero_pd() {
  // DAG-LABEL: test_mm_setzero_pd
  // DAG: store <2 x double> zeroinitializer
  //
  // ASM-LABEL: test_mm_setzero_pd
  // ASM: xorps
  return _mm_setzero_pd();
}

__m128i test_mm_setzero_si128() {
  // DAG-LABEL: test_mm_setzero_si128
  // DAG: store <2 x i64> zeroinitializer
  //
  // ASM-LABEL: test_mm_setzero_si128
  // ASM: xorps
  return _mm_setzero_si128();
}

__m128i test_mm_shuffle_epi32(__m128i A) {
  // DAG-LABEL: test_mm_shuffle_epi32
  // DAG: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  //
  // ASM-LABEL: test_mm_shuffle_epi32
  // ASM: pshufd $0,
  return _mm_shuffle_epi32(A, 0);
}

__m128d test_mm_shuffle_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_shuffle_pd
  // DAG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 2>
  //
  // ASM-LABEL: test_mm_shuffle_pd
  // ASM: shufpd $1,
  return _mm_shuffle_pd(A, B, 1);
}

__m128i test_mm_shufflehi_epi16(__m128i A) {
  // DAG-LABEL: test_mm_shufflehi_epi16
  // DAG: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  //
  // ASM-LABEL: test_mm_shufflehi_epi16
  // ASM: pshufhw $0,
  return _mm_shufflehi_epi16(A, 0);
}

__m128i test_mm_shufflelo_epi16(__m128i A) {
  // DAG-LABEL: test_mm_shufflelo_epi16
  // DAG: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>
  //
  // ASM-LABEL: test_mm_shufflelo_epi16
  // ASM: pshuflw $0,
  return _mm_shufflelo_epi16(A, 0);
}

__m128i test_mm_sll_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sll_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.psll.w
  //
  // ASM-LABEL: test_mm_sll_epi16
  // ASM: psllw
  return _mm_sll_epi16(A, B);
}

__m128i test_mm_sll_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sll_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.psll.d
  //
  // ASM-LABEL: test_mm_sll_epi32
  // ASM: pslld
  return _mm_sll_epi32(A, B);
}

__m128i test_mm_sll_epi64(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sll_epi64
  // DAG: call <2 x i64> @llvm.x86.sse2.psll.q
  //
  // ASM-LABEL: test_mm_sll_epi64
  // ASM: psllq
  return _mm_sll_epi64(A, B);
}

__m128i test_mm_slli_epi16(__m128i A) {
  // DAG-LABEL: test_mm_slli_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.pslli.w
  //
  // ASM-LABEL: test_mm_slli_epi16
  // ASM: psllw
  return _mm_slli_epi16(A, 1);
}

__m128i test_mm_slli_epi32(__m128i A) {
  // DAG-LABEL: test_mm_slli_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.pslli.d
  //
  // ASM-LABEL: test_mm_slli_epi32
  // ASM: pslld
  return _mm_slli_epi32(A, 1);
}

__m128i test_mm_slli_epi64(__m128i A) {
  // DAG-LABEL: test_mm_slli_epi64
  // DAG: call <2 x i64> @llvm.x86.sse2.pslli.q
  //
  // ASM-LABEL: test_mm_slli_epi64
  // ASM: psllq
  return _mm_slli_epi64(A, 1);
}

__m128i test_mm_slli_si128(__m128i A) {
  // DAG-LABEL: test_mm_slli_si128
  // DAG: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  //
  // ASM-LABEL: test_mm_slli_si128
  // ASM: pslldq $5, %xmm{{.*}}
  return _mm_slli_si128(A, 5);
}

__m128d test_mm_sqrt_pd(__m128d A) {
  // DAG-LABEL: test_mm_sqrt_pd
  // DAG: call <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double> %{{.*}})
  //
  // ASM-LABEL: test_mm_sqrt_pd
  // ASM: sqrtpd
  return _mm_sqrt_pd(A);
}

__m128d test_mm_sqrt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_sqrt_sd
  // DAG: call <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double> %{{.*}})
  //
  // ASM-LABEL: test_mm_sqrt_sd
  // ASM: sqrtsd
  return _mm_sqrt_sd(A, B);
}

__m128i test_mm_sra_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sra_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.psra.w
  //
  // ASM-LABEL: test_mm_sra_epi16
  // ASM: psraw
  return _mm_sra_epi16(A, B);
}

__m128i test_mm_sra_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sra_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.psra.d
  //
  // ASM-LABEL: test_mm_sra_epi32
  // ASM: psrad
  return _mm_sra_epi32(A, B);
}

__m128i test_mm_srai_epi16(__m128i A) {
  // DAG-LABEL: test_mm_srai_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.psrai.w
  //
  // ASM-LABEL: test_mm_srai_epi16
  // ASM: psraw
  return _mm_srai_epi16(A, 1);
}

__m128i test_mm_srai_epi32(__m128i A) {
  // DAG-LABEL: test_mm_srai_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.psrai.d
  //
  // ASM-LABEL: test_mm_srai_epi32
  // ASM: psrad
  return _mm_srai_epi32(A, 1);
}

__m128i test_mm_srl_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_srl_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.psrl.w
  //
  // ASM-LABEL: test_mm_srl_epi16
  // ASM: psrlw
  return _mm_srl_epi16(A, B);
}

__m128i test_mm_srl_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_srl_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.psrl.d
  //
  // ASM-LABEL: test_mm_srl_epi32
  // ASM: psrld
  return _mm_srl_epi32(A, B);
}

__m128i test_mm_srl_epi64(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_srl_epi64
  // DAG: call <2 x i64> @llvm.x86.sse2.psrl.q
  //
  // ASM-LABEL: test_mm_srl_epi64
  // ASM: psrlq
  return _mm_srl_epi64(A, B);
}

__m128i test_mm_srli_epi16(__m128i A) {
  // DAG-LABEL: test_mm_srli_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.psrli.w
  //
  // ASM-LABEL: test_mm_srli_epi16
  // ASM: psrlw
  return _mm_srli_epi16(A, 1);
}

__m128i test_mm_srli_epi32(__m128i A) {
  // DAG-LABEL: test_mm_srli_epi32
  // DAG: call <4 x i32> @llvm.x86.sse2.psrli.d
  //
  // ASM-LABEL: test_mm_srli_epi32
  // ASM: psrld
  return _mm_srli_epi32(A, 1);
}

__m128i test_mm_srli_epi64(__m128i A) {
  // DAG-LABEL: test_mm_srli_epi64
  // DAG: call <2 x i64> @llvm.x86.sse2.psrli.q
  //
  // ASM-LABEL: test_mm_srli_epi64
  // ASM: psrlq
  return _mm_srli_epi64(A, 1);
}

__m128i test_mm_srli_si128(__m128i A) {
  // DAG-LABEL: test_mm_srli_si128
  // DAG: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  //
  // ASM-LABEL: test_mm_srli_si128
  // ASM: psrldq $5, %xmm{{.*}}
  return _mm_srli_si128(A, 5);
}

void test_mm_store_pd(double* A, __m128d B) {
  // DAG-LABEL: test_mm_store_pd
  // DAG: store <2 x double> %{{.*}}, <2 x double>* %{{.*}}, align 16
  //
  // ASM-LABEL: test_mm_store_pd
  // ASM: movapd
  _mm_store_pd(A, B);
}

void test_mm_store_sd(double* A, __m128d B) {
  // DAG-LABEL: test_mm_store_sd
  // DAG: store double %{{.*}}, double* %{{.*}}, align 1{{$}}
  //
  // ASM-LABEL: test_mm_store_sd
  // ASM: movsd
  _mm_store_sd(A, B);
}

void test_mm_store_si128(__m128i* A, __m128i B) {
  // DAG-LABEL: test_mm_store_si128
  // DAG: store <2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, align 16
  //
  // ASM-LABEL: test_mm_store_si128
  // ASM: movdqa
  _mm_store_si128(A, B);
}

void test_mm_storeh_pd(double* A, __m128d B) {
  // DAG-LABEL: test_mm_storeh_pd
  // DAG: store double %{{.*}}, double* %{{.*}}, align 1
  //
  // ASM-LABEL: test_mm_storeh_pd
  // ASM: shufpd $1
  // ASM: movsd
  _mm_storeh_pd(A, B);
}

void test_mm_storel_pd(double* A, __m128d B) {
  // DAG-LABEL: test_mm_storel_pd
  // DAG: store double %{{.*}}, double* %{{.*}}, align 1
  //
  // ASM-LABEL: test_mm_storel_pd
  // ASM: movsd
  _mm_storel_pd(A, B);
}

void test_mm_storeu_pd(double* A, __m128d B) {
  // DAG-LABEL: test_mm_storeu_pd
  // DAG: store <2 x double> %{{.*}}, <2 x double>* %{{.*}}, align 1
  //
  // ASM-LABEL: test_mm_storeu_pd
  // ASM: movupd
  _mm_storeu_pd(A, B);
}

void test_mm_storeu_si128(__m128i* A, __m128i B) {
  // DAG-LABEL: test_mm_storeu_si128
  // DAG: store <2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, align 1
  //
  // ASM-LABEL: test_mm_storeu_si128
  // ASM: movdqu
  _mm_storeu_si128(A, B);
}

void test_mm_stream_pd(double *A, __m128d B) {
  // DAG-LABEL: test_mm_stream_pd
  // DAG: store <2 x double> %{{.*}}, <2 x double>* %{{.*}}, align 16, !nontemporal
  //
  // ASM-LABEL: test_mm_stream_pd
  // ASM: movntpd
  _mm_stream_pd(A, B);
}

void test_mm_stream_si32(int *A, int B) {
  // DAG-LABEL: test_mm_stream_si32
  // DAG: store i32 %{{.*}}, i32* %{{.*}}, align 1, !nontemporal
  //
  // ASM-LABEL: test_mm_stream_si32
  // ASM: movntil
  _mm_stream_si32(A, B);
}

void test_mm_stream_si64(long long *A, long long B) {
  // DAG-LABEL: test_mm_stream_si64
  // DAG: store i64 %{{.*}}, i64* %{{.*}}, align 1, !nontemporal
  //
  // ASM-LABEL: test_mm_stream_si64
  // ASM: movntiq
  _mm_stream_si64(A, B);
}

void test_mm_stream_si128(__m128i *A, __m128i B) {
  // DAG-LABEL: test_mm_stream_si128
  // DAG: store <2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, align 16, !nontemporal
  //
  // ASM-LABEL: test_mm_stream_si128
  // ASM: movntdq
  _mm_stream_si128(A, B);
}

__m128i test_mm_sub_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sub_epi8
  // DAG: sub <16 x i8>
  //
  // ASM-LABEL: test_mm_sub_epi8
  // ASM: psubb
  return _mm_sub_epi8(A, B);
}

__m128i test_mm_sub_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sub_epi16
  // DAG: sub <8 x i16>
  //
  // ASM-LABEL: test_mm_sub_epi16
  // ASM: psubw
  return _mm_sub_epi16(A, B);
}

__m128i test_mm_sub_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sub_epi32
  // DAG: sub <4 x i32>
  //
  // ASM-LABEL: test_mm_sub_epi32
  // ASM: psubd
  return _mm_sub_epi32(A, B);
}

__m128i test_mm_sub_epi64(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_sub_epi64
  // DAG: sub <2 x i64>
  //
  // ASM-LABEL: test_mm_sub_epi64
  // ASM: psubq
  return _mm_sub_epi64(A, B);
}

__m128d test_mm_sub_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_sub_pd
  // DAG: fsub <2 x double>
  //
  // ASM-LABEL: test_mm_sub_pd
  // ASM: subpd
  return _mm_sub_pd(A, B);
}

__m128d test_mm_sub_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_sub_sd
  // DAG: fsub double
  //
  // ASM-LABEL: test_mm_sub_sd
  // ASM: subsd
  return _mm_sub_sd(A, B);
}

__m128i test_mm_subs_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_subs_epi8
  // DAG: call <16 x i8> @llvm.x86.sse2.psubs.b
  //
  // ASM-LABEL: test_mm_subs_epi8
  // ASM: psubsb
  return _mm_subs_epi8(A, B);
}

__m128i test_mm_subs_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_subs_epi16
  // DAG: call <8 x i16> @llvm.x86.sse2.psubs.w
  //
  // ASM-LABEL: test_mm_subs_epi16
  // ASM: psubsw
  return _mm_subs_epi16(A, B);
}

__m128i test_mm_subs_epu8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_subs_epu8
  // DAG: call <16 x i8> @llvm.x86.sse2.psubus.b
  //
  // ASM-LABEL: test_mm_subs_epu8
  // ASM: psubusb
  return _mm_subs_epu8(A, B);
}

__m128i test_mm_subs_epu16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_subs_epu16
  // DAG: call <8 x i16> @llvm.x86.sse2.psubus.w
  //
  // ASM-LABEL: test_mm_subs_epu16
  // ASM: psubusw
  return _mm_subs_epu16(A, B);
}

int test_mm_ucomieq_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_ucomieq_sd
  // DAG: call i32 @llvm.x86.sse2.ucomieq.sd
  //
  // ASM-LABEL: test_mm_ucomieq_sd
  // ASM: ucomisd
  return _mm_ucomieq_sd(A, B);
}

int test_mm_ucomige_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_ucomige_sd
  // DAG: call i32 @llvm.x86.sse2.ucomige.sd
  //
  // ASM-LABEL: test_mm_ucomige_sd
  // ASM: ucomisd
  return _mm_ucomige_sd(A, B);
}

int test_mm_ucomigt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_ucomigt_sd
  // DAG: call i32 @llvm.x86.sse2.ucomigt.sd
  //
  // ASM-LABEL: test_mm_ucomigt_sd
  // ASM: ucomisd
  return _mm_ucomigt_sd(A, B);
}

int test_mm_ucomile_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_ucomile_sd
  // DAG: call i32 @llvm.x86.sse2.ucomile.sd
  //
  // ASM-LABEL: test_mm_ucomile_sd
  // ASM: ucomisd
  return _mm_ucomile_sd(A, B);
}

int test_mm_ucomilt_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_ucomilt_sd
  // DAG: call i32 @llvm.x86.sse2.ucomilt.sd
  //
  // ASM-LABEL: test_mm_ucomilt_sd
  // ASM: ucomisd
  return _mm_ucomilt_sd(A, B);
}

int test_mm_ucomineq_sd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_ucomineq_sd
  // DAG: call i32 @llvm.x86.sse2.ucomineq.sd
  //
  // ASM-LABEL: test_mm_ucomineq_sd
  // ASM: ucomisd
  return _mm_ucomineq_sd(A, B);
}

__m128i test_mm_unpackhi_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_unpackhi_epi8
  // DAG: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  //
  // ASM-LABEL: test_mm_unpackhi_epi8
  // ASM: unpckhbw
  return _mm_unpackhi_epi8(A, B);
}

__m128i test_mm_unpackhi_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_unpackhi_epi16
  // DAG: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  //
  // ASM-LABEL: test_mm_unpackhi_epi16
  // ASM: unpckhwd
  return _mm_unpackhi_epi16(A, B);
}

__m128i test_mm_unpackhi_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_unpackhi_epi32
  // DAG: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  //
  // ASM-LABEL: test_mm_unpackhi_epi32
  // ASM: unpckhdq
  return _mm_unpackhi_epi32(A, B);
}

__m128i test_mm_unpackhi_epi64(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_unpackhi_epi64
  // DAG: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 3>
  //
  // ASM-LABEL: test_mm_unpackhi_epi64
  // ASM: unpckhqdq
  return _mm_unpackhi_epi64(A, B);
}

__m128d test_mm_unpackhi_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_unpackhi_pd
  // DAG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  //
  // ASM-LABEL: test_mm_unpackhi_pd
  // ASM: unpckhpd
  return _mm_unpackhi_pd(A, B);
}

__m128i test_mm_unpacklo_epi8(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_unpacklo_epi8
  // DAG: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  //
  // ASM-LABEL: test_mm_unpacklo_epi8
  // ASM: unpcklbw
  return _mm_unpacklo_epi8(A, B);
}

__m128i test_mm_unpacklo_epi16(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_unpacklo_epi16
  // DAG: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  //
  // ASM-LABEL: test_mm_unpacklo_epi16
  // ASM: unpcklwd
  return _mm_unpacklo_epi16(A, B);
}

__m128i test_mm_unpacklo_epi32(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_unpacklo_epi32
  // DAG: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  //
  // ASM-LABEL: test_mm_unpacklo_epi32
  // ASM: unpckldq
  return _mm_unpacklo_epi32(A, B);
}

__m128i test_mm_unpacklo_epi64(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_unpacklo_epi64
  // DAG: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  //
  // ASM-LABEL: test_mm_unpacklo_epi64
  // ASM: unpcklqdq
  return _mm_unpacklo_epi64(A, B);
}

__m128d test_mm_unpacklo_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_unpacklo_pd
  // DAG: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  //
  // ASM-LABEL: test_mm_unpacklo_pd
  // ASM: unpcklpd
  return _mm_unpacklo_pd(A, B);
}

__m128d test_mm_xor_pd(__m128d A, __m128d B) {
  // DAG-LABEL: test_mm_xor_pd
  // DAG: xor <4 x i32> %{{.*}}, %{{.*}}
  //
  // ASM-LABEL: test_mm_xor_pd
  // ASM: pxor
  return _mm_xor_pd(A, B);
}

__m128i test_mm_xor_si128(__m128i A, __m128i B) {
  // DAG-LABEL: test_mm_xor_si128
  // DAG: xor <2 x i64> %{{.*}}, %{{.*}}
  //
  // ASM-LABEL: test_mm_xor_si128
  // ASM: xorps
  return _mm_xor_si128(A, B);
}
