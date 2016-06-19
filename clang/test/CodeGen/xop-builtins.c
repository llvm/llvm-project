// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/xop-intrinsics-fast-isel.ll

__m128i test_mm_maccs_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_maccs_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpmacssww(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_maccs_epi16(a, b, c);
}

__m128i test_mm_macc_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_macc_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpmacsww(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_macc_epi16(a, b, c);
}

__m128i test_mm_maccsd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_maccsd_epi16
  // CHECK: call <4 x i32> @llvm.x86.xop.vpmacsswd(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maccsd_epi16(a, b, c);
}

__m128i test_mm_maccd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_maccd_epi16
  // CHECK: call <4 x i32> @llvm.x86.xop.vpmacswd(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maccd_epi16(a, b, c);
}

__m128i test_mm_maccs_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_maccs_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpmacssdd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maccs_epi32(a, b, c);
}

__m128i test_mm_macc_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_macc_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpmacsdd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_macc_epi32(a, b, c);
}

__m128i test_mm_maccslo_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_maccslo_epi32
  // CHECK: call <2 x i64> @llvm.x86.xop.vpmacssdql(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maccslo_epi32(a, b, c);
}

__m128i test_mm_macclo_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_macclo_epi32
  // CHECK: call <2 x i64> @llvm.x86.xop.vpmacsdql(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_macclo_epi32(a, b, c);
}

__m128i test_mm_maccshi_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_maccshi_epi32
  // CHECK: call <2 x i64> @llvm.x86.xop.vpmacssdqh(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maccshi_epi32(a, b, c);
}

__m128i test_mm_macchi_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_macchi_epi32
  // CHECK: call <2 x i64> @llvm.x86.xop.vpmacsdqh(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_macchi_epi32(a, b, c);
}

__m128i test_mm_maddsd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_maddsd_epi16
  // CHECK: call <4 x i32> @llvm.x86.xop.vpmadcsswd(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maddsd_epi16(a, b, c);
}

__m128i test_mm_maddd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_maddd_epi16
  // CHECK: call <4 x i32> @llvm.x86.xop.vpmadcswd(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maddd_epi16(a, b, c);
}

__m128i test_mm_haddw_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_haddw_epi8
  // CHECK: call <8 x i16> @llvm.x86.xop.vphaddbw(<16 x i8> %{{.*}})
  return _mm_haddw_epi8(a);
}

__m128i test_mm_haddd_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_haddd_epi8
  // CHECK: call <4 x i32> @llvm.x86.xop.vphaddbd(<16 x i8> %{{.*}})
  return _mm_haddd_epi8(a);
}

__m128i test_mm_haddq_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_haddq_epi8
  // CHECK: call <2 x i64> @llvm.x86.xop.vphaddbq(<16 x i8> %{{.*}})
  return _mm_haddq_epi8(a);
}

__m128i test_mm_haddd_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_haddd_epi16
  // CHECK: call <4 x i32> @llvm.x86.xop.vphaddwd(<8 x i16> %{{.*}})
  return _mm_haddd_epi16(a);
}

__m128i test_mm_haddq_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_haddq_epi16
  // CHECK: call <2 x i64> @llvm.x86.xop.vphaddwq(<8 x i16> %{{.*}})
  return _mm_haddq_epi16(a);
}

__m128i test_mm_haddq_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_haddq_epi32
  // CHECK: call <2 x i64> @llvm.x86.xop.vphadddq(<4 x i32> %{{.*}})
  return _mm_haddq_epi32(a);
}

__m128i test_mm_haddw_epu8(__m128i a) {
  // CHECK-LABEL: test_mm_haddw_epu8
  // CHECK: call <8 x i16> @llvm.x86.xop.vphaddubw(<16 x i8> %{{.*}})
  return _mm_haddw_epu8(a);
}

__m128i test_mm_haddd_epu8(__m128i a) {
  // CHECK-LABEL: test_mm_haddd_epu8
  // CHECK: call <4 x i32> @llvm.x86.xop.vphaddubd(<16 x i8> %{{.*}})
  return _mm_haddd_epu8(a);
}

__m128i test_mm_haddq_epu8(__m128i a) {
  // CHECK-LABEL: test_mm_haddq_epu8
  // CHECK: call <2 x i64> @llvm.x86.xop.vphaddubq(<16 x i8> %{{.*}})
  return _mm_haddq_epu8(a);
}

__m128i test_mm_haddd_epu16(__m128i a) {
  // CHECK-LABEL: test_mm_haddd_epu16
  // CHECK: call <4 x i32> @llvm.x86.xop.vphadduwd(<8 x i16> %{{.*}})
  return _mm_haddd_epu16(a);
}

__m128i test_mm_haddq_epu16(__m128i a) {
  // CHECK-LABEL: test_mm_haddq_epu16
  // CHECK: call <2 x i64> @llvm.x86.xop.vphadduwq(<8 x i16> %{{.*}})
  return _mm_haddq_epu16(a);
}

__m128i test_mm_haddq_epu32(__m128i a) {
  // CHECK-LABEL: test_mm_haddq_epu32
  // CHECK: call <2 x i64> @llvm.x86.xop.vphaddudq(<4 x i32> %{{.*}})
  return _mm_haddq_epu32(a);
}

__m128i test_mm_hsubw_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_hsubw_epi8
  // CHECK: call <8 x i16> @llvm.x86.xop.vphsubbw(<16 x i8> %{{.*}})
  return _mm_hsubw_epi8(a);
}

__m128i test_mm_hsubd_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_hsubd_epi16
  // CHECK: call <4 x i32> @llvm.x86.xop.vphsubwd(<8 x i16> %{{.*}})
  return _mm_hsubd_epi16(a);
}

__m128i test_mm_hsubq_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_hsubq_epi32
  // CHECK: call <2 x i64> @llvm.x86.xop.vphsubdq(<4 x i32> %{{.*}})
  return _mm_hsubq_epi32(a);
}

__m128i test_mm_cmov_si128(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_cmov_si128
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcmov(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_cmov_si128(a, b, c);
}

__m256i test_mm256_cmov_si256(__m256i a, __m256i b, __m256i c) {
  // CHECK-LABEL: test_mm256_cmov_si256
  // CHECK: call <4 x i64> @llvm.x86.xop.vpcmov.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_cmov_si256(a, b, c);
}

__m128i test_mm_perm_epi8(__m128i a, __m128i b, __m128i c) {
  // CHECK-LABEL: test_mm_perm_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpperm(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_perm_epi8(a, b, c);
}

__m128i test_mm_rot_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_rot_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vprotb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_rot_epi8(a, b);
}

__m128i test_mm_rot_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_rot_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vprotw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_rot_epi16(a, b);
}

__m128i test_mm_rot_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_rot_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vprotd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_rot_epi32(a, b);
}

__m128i test_mm_rot_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_rot_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vprotq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_rot_epi64(a, b);
}

__m128i test_mm_roti_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_roti_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vprotbi(<16 x i8> %{{.*}}, i8 1)
  return _mm_roti_epi8(a, 1);
}

__m128i test_mm_roti_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_roti_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vprotwi(<8 x i16> %{{.*}}, i8 50)
  return _mm_roti_epi16(a, 50);
}

__m128i test_mm_roti_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_roti_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vprotdi(<4 x i32> %{{.*}}, i8 -30)
  return _mm_roti_epi32(a, -30);
}

__m128i test_mm_roti_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_roti_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vprotqi(<2 x i64> %{{.*}}, i8 100)
  return _mm_roti_epi64(a, 100);
}

__m128i test_mm_shl_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_shl_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpshlb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_shl_epi8(a, b);
}

__m128i test_mm_shl_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_shl_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpshlw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_shl_epi16(a, b);
}

__m128i test_mm_shl_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_shl_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpshld(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_shl_epi32(a, b);
}

__m128i test_mm_shl_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_shl_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpshlq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_shl_epi64(a, b);
}

__m128i test_mm_sha_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sha_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpshab(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_sha_epi8(a, b);
}

__m128i test_mm_sha_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sha_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpshaw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_sha_epi16(a, b);
}

__m128i test_mm_sha_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sha_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpshad(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sha_epi32(a, b);
}

__m128i test_mm_sha_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_sha_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpshaq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_sha_epi64(a, b);
}

__m128i test_mm_com_epu8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_com_epu8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomub(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 0)
  return _mm_com_epu8(a, b, 0);
}

__m128i test_mm_com_epu16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_com_epu16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomuw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 0)
  return _mm_com_epu16(a, b, 0);
}

__m128i test_mm_com_epu32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_com_epu32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomud(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 0)
  return _mm_com_epu32(a, b, 0);
}

__m128i test_mm_com_epu64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_com_epu64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomuq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 0)
  return _mm_com_epu64(a, b, 0);
}

__m128i test_mm_com_epi8(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_com_epi8
  // CHECK: call <16 x i8> @llvm.x86.xop.vpcomb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 0)
  return _mm_com_epi8(a, b, 0);
}

__m128i test_mm_com_epi16(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_com_epi16
  // CHECK: call <8 x i16> @llvm.x86.xop.vpcomw(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, i8 0)
  return _mm_com_epi16(a, b, 0);
}

__m128i test_mm_com_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_com_epi32
  // CHECK: call <4 x i32> @llvm.x86.xop.vpcomd(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i8 0)
  return _mm_com_epi32(a, b, 0);
}

__m128i test_mm_com_epi64(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_com_epi64
  // CHECK: call <2 x i64> @llvm.x86.xop.vpcomq(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i8 0)
  return _mm_com_epi64(a, b, 0);
}

__m128d test_mm_permute2_pd(__m128d a, __m128d b, __m128i c) {
  // CHECK-LABEL: test_mm_permute2_pd
  // CHECK: call <2 x double> @llvm.x86.xop.vpermil2pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i64> %{{.*}}, i8 0)
  return _mm_permute2_pd(a, b, c, 0);
}

__m256d test_mm256_permute2_pd(__m256d a, __m256d b, __m256i c) {
  // CHECK-LABEL: test_mm256_permute2_pd
  // CHECK: call <4 x double> @llvm.x86.xop.vpermil2pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i64> %{{.*}}, i8 0)
  return _mm256_permute2_pd(a, b, c, 0);
}

__m128 test_mm_permute2_ps(__m128 a, __m128 b, __m128i c) {
  // CHECK-LABEL: test_mm_permute2_ps
  // CHECK: call <4 x float> @llvm.x86.xop.vpermil2ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> %{{.*}}, i8 0)
  return _mm_permute2_ps(a, b, c, 0);
}

__m256 test_mm256_permute2_ps(__m256 a, __m256 b, __m256i c) {
  // CHECK-LABEL: test_mm256_permute2_ps
  // CHECK: call <8 x float> @llvm.x86.xop.vpermil2ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> %{{.*}}, i8 0)
  return _mm256_permute2_ps(a, b, c, 0);
}

__m128 test_mm_frcz_ss(__m128 a) {
  // CHECK-LABEL: test_mm_frcz_ss
  // CHECK: call <4 x float> @llvm.x86.xop.vfrcz.ss(<4 x float> %{{.*}})
  return _mm_frcz_ss(a);
}

__m128d test_mm_frcz_sd(__m128d a) {
  // CHECK-LABEL: test_mm_frcz_sd
  // CHECK: call <2 x double> @llvm.x86.xop.vfrcz.sd(<2 x double> %{{.*}})
  return _mm_frcz_sd(a);
}

__m128 test_mm_frcz_ps(__m128 a) {
  // CHECK-LABEL: test_mm_frcz_ps
  // CHECK: call <4 x float> @llvm.x86.xop.vfrcz.ps(<4 x float> %{{.*}})
  return _mm_frcz_ps(a);
}

__m128d test_mm_frcz_pd(__m128d a) {
  // CHECK-LABEL: test_mm_frcz_pd
  // CHECK: call <2 x double> @llvm.x86.xop.vfrcz.pd(<2 x double> %{{.*}})
  return _mm_frcz_pd(a);
}

__m256 test_mm256_frcz_ps(__m256 a) {
  // CHECK-LABEL: test_mm256_frcz_ps
  // CHECK: call <8 x float> @llvm.x86.xop.vfrcz.ps.256(<8 x float> %{{.*}})
  return _mm256_frcz_ps(a);
}

__m256d test_mm256_frcz_pd(__m256d a) {
  // CHECK-LABEL: test_mm256_frcz_pd
  // CHECK: call <4 x double> @llvm.x86.xop.vfrcz.pd.256(<4 x double> %{{.*}})
  return _mm256_frcz_pd(a);
}
