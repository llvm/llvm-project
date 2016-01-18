// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128i test_blend_epi16(__m128i V1, __m128i V2) {
  // CHECK-LABEL: test_blend_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>
  // CHECK-ASM: pblendw $42, %xmm{{.*}}, %xmm{{.*}}
  return _mm_blend_epi16(V1, V2, 42);
}

__m128d test_blend_pd(__m128d V1, __m128d V2) {
  // CHECK-LABEL: test_blend_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>
  // CHECK-ASM: blendpd $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_blend_pd(V1, V2, 2);
}

__m128 test_blend_ps(__m128 V1, __m128 V2) {
  // CHECK-LABEL: test_blend_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  // CHECK-ASM: blendps $6, %xmm{{.*}}, %xmm{{.*}}
  return _mm_blend_ps(V1, V2, 6);
}

__m128i test_blendv_epi8(__m128i V1, __m128i V2, __m128i V3) {
  // CHECK-LABEL: test_blendv_epi8
  // CHECK: call <16 x i8> @llvm.x86.sse41.pblendvb
  // CHECK-ASM: pblendvb %xmm{{.*}}, %xmm{{.*}}
  return _mm_blendv_epi8(V1, V2, V3);
}

__m128d test_blendv_pd(__m128d V1, __m128d V2, __m128d V3) {
  // CHECK-LABEL: test_blendv_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.blendvpd
  // CHECK-ASM: blendvpd %xmm{{.*}}, %xmm{{.*}}
  return _mm_blendv_pd(V1, V2, V3);
}

__m128 test_blendv_ps(__m128 V1, __m128 V2, __m128 V3) {
  // CHECK-LABEL: test_blendv_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.blendvps
  // CHECK-ASM: blendvps %xmm{{.*}}, %xmm{{.*}}
  return _mm_blendv_ps(V1, V2, V3);
}

__m128d test_mm_ceil_pd(__m128d x) {
  // CHECK-LABEL: test_mm_ceil_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.pd
  // CHECK-ASM: roundpd $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_ceil_pd(x);
}

__m128 test_mm_ceil_ps(__m128 x) {
  // CHECK-LABEL: test_mm_ceil_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ps
  // CHECK-ASM: roundps $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_ceil_ps(x);
}

__m128d test_mm_ceil_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_ceil_sd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.sd
  // CHECK-ASM: roundsd $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_ceil_sd(x, y);
}

__m128 test_mm_ceil_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_ceil_ss
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ss
  // CHECK-ASM: roundss $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_ceil_ss(x, y);
}

__m128i test_mm_cmpeq_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpeq_epi64
  // CHECK: icmp eq <2 x i64>
  // CHECK-ASM: pcmpeqq %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpeq_epi64(A, B);
}

__m128i test_mm_cvtepi8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi16
  // CHECK: sext <8 x i8> {{.*}} to <8 x i16>
  // CHECK-ASM: pmovsxbw %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepi8_epi16(a);
}

__m128i test_mm_cvtepi8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi32
  // CHECK: sext <4 x i8> {{.*}} to <4 x i32>
  // CHECK-ASM: pmovsxbd %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepi8_epi32(a);
}

__m128i test_mm_cvtepi8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi64
  // CHECK: sext <2 x i8> {{.*}} to <2 x i64>
  // CHECK-ASM: pmovsxbq %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepi8_epi64(a);
}

__m128i test_mm_cvtepi16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi16_epi32
  // CHECK: sext <4 x i16> {{.*}} to <4 x i32>
  // CHECK-ASM: pmovsxwd %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepi16_epi32(a);
}

__m128i test_mm_cvtepi16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi16_epi64
  // CHECK: sext <2 x i16> {{.*}} to <2 x i64>
  // CHECK-ASM: pmovsxwq %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepi16_epi64(a);
}

__m128i test_mm_cvtepi32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi32_epi64
  // CHECK: sext <2 x i32> {{.*}} to <2 x i64>
  // CHECK-ASM: pmovsxdq %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepi32_epi64(a);
}

__m128i test_mm_cvtepu8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi16
  // CHECK: call <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8> {{.*}})
  // CHECK-ASM: pmovzxbw %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepu8_epi16(a);
}

__m128i test_mm_cvtepu8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse41.pmovzxbd(<16 x i8> {{.*}})
  // CHECK-ASM: pmovzxbd %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepu8_epi32(a);
}

__m128i test_mm_cvtepu8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi64
  // CHECK: call <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8> {{.*}})
  // CHECK-ASM: pmovzxbq %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepu8_epi64(a);
}

__m128i test_mm_cvtepu16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu16_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16> {{.*}})
  // CHECK-ASM: pmovzxwd %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepu16_epi32(a);
}

__m128i test_mm_cvtepu16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu16_epi64
  // CHECK: call <2 x i64> @llvm.x86.sse41.pmovzxwq(<8 x i16> {{.*}})
  // CHECK-ASM: pmovzxwq %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepu16_epi64(a);
}

__m128i test_mm_cvtepu32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu32_epi64
  // CHECK: call <2 x i64> @llvm.x86.sse41.pmovzxdq(<4 x i32> {{.*}})
  // CHECK-ASM: pmovzxdq %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtepu32_epi64(a);
}

__m128d test_mm_dp_pd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_dp_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.dppd
  // CHECK-ASM: dppd $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_dp_pd(x, y, 2);
}

__m128 test_mm_dp_ps(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_dp_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.dpps
  // CHECK-ASM: dpps $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_dp_ps(x, y, 2);
}

int test_extract_epi8(__m128i x) {
  // CHECK-LABEL: test_extract_epi8
  // CHECK: extractelement <16 x i8> %{{.*}}, i32 0
  // CHECK-ASM: pextrb
  return _mm_extract_epi8(x, 16);
}

int test_extract_epi32(__m128i x) {
  // CHECK-LABEL: test_extract_epi32
  // CHECK: extractelement <4 x i32> %{{.*}}, i32 1
  // CHECK-ASM: pextrd
  return _mm_extract_epi32(x, 1);
}

long long test_extract_epi64(__m128i x) {
  // CHECK-LABEL: test_extract_epi64
  // CHECK: extractelement <2 x i64> %{{.*}}, i32 1
  // CHECK-ASM: pextrq
  return _mm_extract_epi64(x, 1);
}

//TODO
//int test_extract_ps(__m128i x) {
//  return _mm_extract_ps(_mm_add_ps(x,x), 1);
//}

__m128d test_mm_floor_pd(__m128d x) {
  // CHECK-LABEL: test_mm_floor_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.pd
  // CHECK-ASM: roundpd $1, %xmm{{.*}}, %xmm{{.*}}
  return _mm_floor_pd(x);
}

__m128 test_mm_floor_ps(__m128 x) {
  // CHECK-LABEL: test_mm_floor_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ps
  // CHECK-ASM: roundps $1, %xmm{{.*}}, %xmm{{.*}}
  return _mm_floor_ps(x);
}

__m128d test_mm_floor_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_floor_sd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.sd
  // CHECK-ASM: roundsd $1, %xmm{{.*}}, %xmm{{.*}}
  return _mm_floor_sd(x, y);
}

__m128 test_mm_floor_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_floor_ss
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ss
  // CHECK-ASM: roundss $1, %xmm{{.*}}, %xmm{{.*}}
  return _mm_floor_ss(x, y);
}

__m128i test_insert_epi8(__m128i x, char b) {
  // CHECK-LABEL: test_insert_epi8
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 0
  // CHECK-ASM: pinsrb
  return _mm_insert_epi8(x, b, 16);
}

__m128i test_insert_epi32(__m128i x, int b) {
  // CHECK-LABEL: test_insert_epi32
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 0
  // CHECK-ASM: pinsrd
  return _mm_insert_epi32(x, b, 4);
}

__m128i test_insert_epi64(__m128i x, long long b) {
  // CHECK-LABEL: test_insert_epi64
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 0
  // CHECK-ASM: pinsrq
  return _mm_insert_epi64(x, b, 2);
}

__m128 test_insert_ps(__m128 x, __m128 y) {
  // CHECK-LABEL: test_insert_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.insertps
  // CHECK-ASM: insertps $5, %xmm{{.*}}, %xmm{{.*}}
  return _mm_insert_ps(x, y, 5);
}

__m128i test_mm_max_epi8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epi8
  // CHECK: call <16 x i8> @llvm.x86.sse41.pmaxsb
  // CHECK-ASM: pmaxsb %xmm{{.*}}, %xmm{{.*}}
  return _mm_max_epi8(x, y);
}

__m128i test_mm_max_epu16(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epu16
  // CHECK: call <8 x i16> @llvm.x86.sse41.pmaxuw
  // CHECK-ASM: pmaxuw %xmm{{.*}}, %xmm{{.*}}
  return _mm_max_epu16(x, y);
}

__m128i test_mm_max_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse41.pmaxsd
  // CHECK-ASM: pmaxsd %xmm{{.*}}, %xmm{{.*}}
  return _mm_max_epi32(x, y);
}

__m128i test_mm_max_epu32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epu32
  // CHECK: call <4 x i32> @llvm.x86.sse41.pmaxud
  // CHECK-ASM: pmaxud %xmm{{.*}}, %xmm{{.*}}
  return _mm_max_epu32(x, y);
}

__m128i test_mm_min_epi8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epi8
  // CHECK: call <16 x i8> @llvm.x86.sse41.pminsb
  // CHECK-ASM: pminsb %xmm{{.*}}, %xmm{{.*}}
  return _mm_min_epi8(x, y);
}

__m128i test_mm_min_epu16(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epu16
  // CHECK: call <8 x i16> @llvm.x86.sse41.pminuw
  // CHECK-ASM: pminuw %xmm{{.*}}, %xmm{{.*}}
  return _mm_min_epu16(x, y);
}

__m128i test_mm_min_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epi32
  // CHECK: call <4 x i32> @llvm.x86.sse41.pminsd
  // CHECK-ASM: pminsd %xmm{{.*}}, %xmm{{.*}}
  return _mm_min_epi32(x, y);
}

__m128i test_mm_min_epu32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epu32
  // CHECK: call <4 x i32> @llvm.x86.sse41.pminud
  // CHECK-ASM: pminud %xmm{{.*}}, %xmm{{.*}}
  return _mm_min_epu32(x, y);
}

__m128i test_mm_minpos_epu16(__m128i x) {
  // CHECK-LABEL: test_mm_minpos_epu16
  // CHECK: call <8 x i16> @llvm.x86.sse41.phminposuw
  // CHECK-ASM: phminposuw %xmm{{.*}}, %xmm{{.*}}
  return _mm_minpos_epu16(x);
}

__m128i test_mm_mpsadbw_epu8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mpsadbw_epu8
  // CHECK: call <8 x i16> @llvm.x86.sse41.mpsadbw
  // CHECK-ASM: mpsadbw $1, %xmm{{.*}}, %xmm{{.*}}
  return _mm_mpsadbw_epu8(x, y, 1);
}

__m128i test_mm_mul_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mul_epi32
  // CHECK: call <2 x i64> @llvm.x86.sse41.pmuldq
  // CHECK-ASM: pmuldq %xmm{{.*}}, %xmm{{.*}}
  return _mm_mul_epi32(x, y);
}

__m128i test_mm_mullo_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mullo_epi32
  // CHECK: mul <4 x i32>
  // CHECK-ASM: pmulld %xmm{{.*}}, %xmm{{.*}}
  return _mm_mullo_epi32(x, y);
}

__m128i test_mm_packus_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_packus_epi32
  // CHECK: call <8 x i16> @llvm.x86.sse41.packusdw
  // CHECK-ASM: packusdw %xmm{{.*}}, %xmm{{.*}}
  return _mm_packus_epi32(x, y);
}

__m128d test_mm_round_pd(__m128d x) {
  // CHECK-LABEL: test_mm_round_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.pd
  // CHECK-ASM: roundpd $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_round_pd(x, 2);
}

__m128 test_mm_round_ps(__m128 x) {
  // CHECK-LABEL: test_mm_round_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ps
  // CHECK-ASM: roundps $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_round_ps(x, 2);
}

__m128d test_mm_round_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_round_sd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.sd
  // CHECK-ASM: roundsd $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_round_sd(x, y, 2);
}

__m128 test_mm_round_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_round_ss
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ss
  // CHECK-ASM: roundss $2, %xmm{{.*}}, %xmm{{.*}}
  return _mm_round_ss(x, y, 2);
}

__m128i test_mm_stream_load_si128(__m128i const *a) {
  // CHECK-LABEL: test_mm_stream_load_si128
  // CHECK: call <2 x i64> @llvm.x86.sse41.movntdqa
  // CHECK-ASM: movntdqa
  return _mm_stream_load_si128(a);
}

int test_mm_test_all_ones(__m128i x) {
  // CHECK-LABEL: test_mm_test_all_ones
  // CHECK: call i32 @llvm.x86.sse41.ptestc
  // CHECK-ASM: ptest %xmm{{.*}}, %xmm{{.*}}
  return _mm_test_all_ones(x);
}

int test_mm_test_all_zeros(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_test_all_zeros
  // CHECK: call i32 @llvm.x86.sse41.ptestz
  // CHECK-ASM: ptest %xmm{{.*}}, %xmm{{.*}}
  return _mm_test_all_zeros(x, y);
}

int test_mm_test_mix_ones_zeros(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_test_mix_ones_zeros
  // CHECK: call i32 @llvm.x86.sse41.ptestnzc
  // CHECK-ASM: ptest %xmm{{.*}}, %xmm{{.*}}
  return _mm_test_mix_ones_zeros(x, y);
}

int test_mm_testc_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testc_si128
  // CHECK: call i32 @llvm.x86.sse41.ptestc
  // CHECK-ASM: ptest %xmm{{.*}}, %xmm{{.*}}
  return _mm_testc_si128(x, y);
}

int test_mm_testnzc_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testnzc_si128
  // CHECK: call i32 @llvm.x86.sse41.ptestnzc
  // CHECK-ASM: ptest %xmm{{.*}}, %xmm{{.*}}
  return _mm_testnzc_si128(x, y);
}

int test_mm_testz_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testz_si128
  // CHECK: call i32 @llvm.x86.sse41.ptestz
  // CHECK-ASM: ptest %xmm{{.*}}, %xmm{{.*}}
  return _mm_testz_si128(x, y);
}
