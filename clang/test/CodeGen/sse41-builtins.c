// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/sse41-intrinsics-fast-isel.ll

__m128i test_mm_blend_epi16(__m128i V1, __m128i V2) {
  // CHECK-LABEL: test_mm_blend_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 7>
  return _mm_blend_epi16(V1, V2, 42);
}

__m128d test_mm_blend_pd(__m128d V1, __m128d V2) {
  // CHECK-LABEL: test_mm_blend_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_blend_pd(V1, V2, 2);
}

__m128 test_mm_blend_ps(__m128 V1, __m128 V2) {
  // CHECK-LABEL: test_mm_blend_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  return _mm_blend_ps(V1, V2, 6);
}

__m128i test_mm_blendv_epi8(__m128i V1, __m128i V2, __m128i V3) {
  // CHECK-LABEL: test_mm_blendv_epi8
  // CHECK: call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_blendv_epi8(V1, V2, V3);
}

__m128d test_mm_blendv_pd(__m128d V1, __m128d V2, __m128d V3) {
  // CHECK-LABEL: test_mm_blendv_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_blendv_pd(V1, V2, V3);
}

__m128 test_mm_blendv_ps(__m128 V1, __m128 V2, __m128 V3) {
  // CHECK-LABEL: test_mm_blendv_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_blendv_ps(V1, V2, V3);
}

__m128d test_mm_ceil_pd(__m128d x) {
  // CHECK-LABEL: test_mm_ceil_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 2)
  return _mm_ceil_pd(x);
}

__m128 test_mm_ceil_ps(__m128 x) {
  // CHECK-LABEL: test_mm_ceil_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 2)
  return _mm_ceil_ps(x);
}

__m128d test_mm_ceil_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_ceil_sd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 2)
  return _mm_ceil_sd(x, y);
}

__m128 test_mm_ceil_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_ceil_ss
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 2)
  return _mm_ceil_ss(x, y);
}

__m128i test_mm_cmpeq_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpeq_epi64
  // CHECK: icmp eq <2 x i64>
  // CHECK: sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_cmpeq_epi64(A, B);
}

__m128i test_mm_cvtepi8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi16
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: sext <8 x i8> {{.*}} to <8 x i16>
  return _mm_cvtepi8_epi16(a);
}

__m128i test_mm_cvtepi8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi32
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sext <4 x i8> {{.*}} to <4 x i32>
  return _mm_cvtepi8_epi32(a);
}

__m128i test_mm_cvtepi8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi8_epi64
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sext <2 x i8> {{.*}} to <2 x i64>
  return _mm_cvtepi8_epi64(a);
}

__m128i test_mm_cvtepi16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi16_epi32
  // CHECK: shufflevector <8 x i16> {{.*}}, <8 x i16> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: sext <4 x i16> {{.*}} to <4 x i32>
  return _mm_cvtepi16_epi32(a);
}

__m128i test_mm_cvtepi16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi16_epi64
  // CHECK: shufflevector <8 x i16> {{.*}}, <8 x i16> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sext <2 x i16> {{.*}} to <2 x i64>
  return _mm_cvtepi16_epi64(a);
}

__m128i test_mm_cvtepi32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepi32_epi64
  // CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: sext <2 x i32> {{.*}} to <2 x i64>
  return _mm_cvtepi32_epi64(a);
}

__m128i test_mm_cvtepu8_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi16
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: zext <8 x i8> {{.*}} to <8 x i16>
  return _mm_cvtepu8_epi16(a);
}

__m128i test_mm_cvtepu8_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi32
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: zext <4 x i8> {{.*}} to <4 x i32>
  return _mm_cvtepu8_epi32(a);
}

__m128i test_mm_cvtepu8_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu8_epi64
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: zext <2 x i8> {{.*}} to <2 x i64>
  return _mm_cvtepu8_epi64(a);
}

__m128i test_mm_cvtepu16_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu16_epi32
  // CHECK: shufflevector <8 x i16> {{.*}}, <8 x i16> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: zext <4 x i16> {{.*}} to <4 x i32>
  return _mm_cvtepu16_epi32(a);
}

__m128i test_mm_cvtepu16_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu16_epi64
  // CHECK: shufflevector <8 x i16> {{.*}}, <8 x i16> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: zext <2 x i16> {{.*}} to <2 x i64>
  return _mm_cvtepu16_epi64(a);
}

__m128i test_mm_cvtepu32_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_cvtepu32_epi64
  // CHECK: shufflevector <4 x i32> {{.*}}, <4 x i32> {{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: zext <2 x i32> {{.*}} to <2 x i64>
  return _mm_cvtepu32_epi64(a);
}

__m128d test_mm_dp_pd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_dp_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.dppd(<2 x double> {{.*}}, <2 x double> {{.*}}, i8 7)
  return _mm_dp_pd(x, y, 7);
}

__m128 test_mm_dp_ps(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_dp_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.dpps(<4 x float> {{.*}}, <4 x float> {{.*}}, i8 7)
  return _mm_dp_ps(x, y, 7);
}

int test_mm_extract_epi8(__m128i x) {
  // CHECK-LABEL: test_mm_extract_epi8
  // CHECK: extractelement <16 x i8> %{{.*}}, {{i32|i64}} 1
  // CHECK: zext i8 %{{.*}} to i32
  return _mm_extract_epi8(x, 1);
}

int test_mm_extract_epi32(__m128i x) {
  // CHECK-LABEL: test_mm_extract_epi32
  // CHECK: extractelement <4 x i32> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_epi32(x, 1);
}

long long test_mm_extract_epi64(__m128i x) {
  // CHECK-LABEL: test_mm_extract_epi64
  // CHECK: extractelement <2 x i64> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_epi64(x, 1);
}

int test_mm_extract_ps(__m128 x) {
  // CHECK-LABEL: test_mm_extract_ps
  // CHECK: extractelement <4 x float> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_ps(x, 1);
}

__m128d test_mm_floor_pd(__m128d x) {
  // CHECK-LABEL: test_mm_floor_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 1)
  return _mm_floor_pd(x);
}

__m128 test_mm_floor_ps(__m128 x) {
  // CHECK-LABEL: test_mm_floor_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 1)
  return _mm_floor_ps(x);
}

__m128d test_mm_floor_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_floor_sd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 1)
  return _mm_floor_sd(x, y);
}

__m128 test_mm_floor_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_floor_ss
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 1)
  return _mm_floor_ss(x, y);
}

__m128i test_mm_insert_epi8(__m128i x, char b) {
  // CHECK-LABEL: test_mm_insert_epi8
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi8(x, b, 1);
}

__m128i test_mm_insert_epi32(__m128i x, int b) {
  // CHECK-LABEL: test_mm_insert_epi32
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi32(x, b, 1);
}

__m128i test_mm_insert_epi64(__m128i x, long long b) {
  // CHECK-LABEL: test_mm_insert_epi64
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi64(x, b, 1);
}

__m128 test_mm_insert_ps(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_insert_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 4)
  return _mm_insert_ps(x, y, 4);
}

__m128i test_mm_max_epi8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epi8
  // CHECK:       [[CMP:%.*]] = icmp sgt <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  return _mm_max_epi8(x, y);
}

__m128i test_mm_max_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epi32
  // CHECK:       [[CMP:%.*]] = icmp sgt <4 x i32> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <4 x i1> [[CMP]], <4 x i32> [[X]], <4 x i32> [[Y]]
  return _mm_max_epi32(x, y);
}

__m128i test_mm_max_epu16(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epu16
  // CHECK:       [[CMP:%.*]] = icmp ugt <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  return _mm_max_epu16(x, y);
}

__m128i test_mm_max_epu32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_max_epu32
  // CHECK:       [[CMP:%.*]] = icmp ugt <4 x i32> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <4 x i1> [[CMP]], <4 x i32> [[X]], <4 x i32> [[Y]]
  return _mm_max_epu32(x, y);
}

__m128i test_mm_min_epi8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epi8
  // CHECK:       [[CMP:%.*]] = icmp slt <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  return _mm_min_epi8(x, y);
}

__m128i test_mm_min_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epi32
  // CHECK:       [[CMP:%.*]] = icmp slt <4 x i32> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <4 x i1> [[CMP]], <4 x i32> [[X]], <4 x i32> [[Y]]
  return _mm_min_epi32(x, y);
}

__m128i test_mm_min_epu16(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epu16
  // CHECK:       [[CMP:%.*]] = icmp ult <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  return _mm_min_epu16(x, y);
}

__m128i test_mm_min_epu32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_min_epu32
  // CHECK:       [[CMP:%.*]] = icmp ult <4 x i32> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  select <4 x i1> [[CMP]], <4 x i32> [[X]], <4 x i32> [[Y]]
  return _mm_min_epu32(x, y);
}

__m128i test_mm_minpos_epu16(__m128i x) {
  // CHECK-LABEL: test_mm_minpos_epu16
  // CHECK: call <8 x i16> @llvm.x86.sse41.phminposuw(<8 x i16> %{{.*}})
  return _mm_minpos_epu16(x);
}

__m128i test_mm_mpsadbw_epu8(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mpsadbw_epu8
  // CHECK: call <8 x i16> @llvm.x86.sse41.mpsadbw(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 1)
  return _mm_mpsadbw_epu8(x, y, 1);
}

__m128i test_mm_mul_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mul_epi32
  // CHECK: shl <2 x i64> %{{.*}}, <i64 32, i64 32>
  // CHECK: ashr <2 x i64> %{{.*}}, <i64 32, i64 32>
  // CHECK: shl <2 x i64> %{{.*}}, <i64 32, i64 32>
  // CHECK: ashr <2 x i64> %{{.*}}, <i64 32, i64 32>
  // CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mul_epi32(x, y);
}

__m128i test_mm_mullo_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_mullo_epi32
  // CHECK: mul <4 x i32>
  return _mm_mullo_epi32(x, y);
}

__m128i test_mm_packus_epi32(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_packus_epi32
  // CHECK: call <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_packus_epi32(x, y);
}

__m128d test_mm_round_pd(__m128d x) {
  // CHECK-LABEL: test_mm_round_pd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 4)
  return _mm_round_pd(x, 4);
}

__m128 test_mm_round_ps(__m128 x) {
  // CHECK-LABEL: test_mm_round_ps
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 4)
  return _mm_round_ps(x, 4);
}

__m128d test_mm_round_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_round_sd
  // CHECK: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 4)
  return _mm_round_sd(x, y, 4);
}

__m128 test_mm_round_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_round_ss
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 4)
  return _mm_round_ss(x, y, 4);
}

__m128i test_mm_stream_load_si128(__m128i const *a) {
  // CHECK-LABEL: test_mm_stream_load_si128
  // CHECK: load <2 x i64>, <2 x i64>* %{{.*}}, align 16, !nontemporal
  return _mm_stream_load_si128(a);
}

int test_mm_test_all_ones(__m128i x) {
  // CHECK-LABEL: test_mm_test_all_ones
  // CHECK: call i32 @llvm.x86.sse41.ptestc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_test_all_ones(x);
}

int test_mm_test_all_zeros(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_test_all_zeros
  // CHECK: call i32 @llvm.x86.sse41.ptestz(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_test_all_zeros(x, y);
}

int test_mm_test_mix_ones_zeros(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_test_mix_ones_zeros
  // CHECK: call i32 @llvm.x86.sse41.ptestnzc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_test_mix_ones_zeros(x, y);
}

int test_mm_testc_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testc_si128
  // CHECK: call i32 @llvm.x86.sse41.ptestc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_testc_si128(x, y);
}

int test_mm_testnzc_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testnzc_si128
  // CHECK: call i32 @llvm.x86.sse41.ptestnzc(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_testnzc_si128(x, y);
}

int test_mm_testz_si128(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_testz_si128
  // CHECK: call i32 @llvm.x86.sse41.ptestz(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_testz_si128(x, y);
}
