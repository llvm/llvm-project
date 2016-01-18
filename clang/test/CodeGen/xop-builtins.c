// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +xop -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128i test_mm_maccs_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacssww
  // CHECK-ASM: vpmacssww %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maccs_epi16(a, b, c);
}

__m128i test_mm_macc_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsww
  // CHECK-ASM: vpmacsww %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_macc_epi16(a, b, c);
}

__m128i test_mm_maccsd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsswd
  // CHECK-ASM: vpmacsswd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maccsd_epi16(a, b, c);
}

__m128i test_mm_maccd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacswd
  // CHECK-ASM: vpmacswd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maccd_epi16(a, b, c);
}

__m128i test_mm_maccs_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacssdd
  // CHECK-ASM: vpmacssdd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maccs_epi32(a, b, c);
}

__m128i test_mm_macc_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsdd
  // CHECK-ASM: vpmacsdd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_macc_epi32(a, b, c);
}

__m128i test_mm_maccslo_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacssdql
  // CHECK-ASM: vpmacssdql %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maccslo_epi32(a, b, c);
}

__m128i test_mm_macclo_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsdql
  // CHECK-ASM: vpmacsdql %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_macclo_epi32(a, b, c);
}

__m128i test_mm_maccshi_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacssdqh
  // CHECK-ASM: vpmacssdqh %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maccshi_epi32(a, b, c);
}

__m128i test_mm_macchi_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsdqh
  // CHECK-ASM: vpmacsdqh %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_macchi_epi32(a, b, c);
}

__m128i test_mm_maddsd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmadcsswd
  // CHECK-ASM: vpmadcsswd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maddsd_epi16(a, b, c);
}

__m128i test_mm_maddd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmadcswd
  // CHECK-ASM: vpmadcswd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maddd_epi16(a, b, c);
}

__m128i test_mm_haddw_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddbw
  // CHECK-ASM: vphaddbw %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddw_epi8(a);
}

__m128i test_mm_haddd_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddbd
  // CHECK-ASM: vphaddbd %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddd_epi8(a);
}

__m128i test_mm_haddq_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddbq
  // CHECK-ASM: vphaddbq %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddq_epi8(a);
}

__m128i test_mm_haddd_epi16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddwd
  // CHECK-ASM: vphaddwd %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddd_epi16(a);
}

__m128i test_mm_haddq_epi16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddwq
  // CHECK-ASM: vphaddwq %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddq_epi16(a);
}

__m128i test_mm_haddq_epi32(__m128i a) {
  // CHECK: @llvm.x86.xop.vphadddq
  // CHECK-ASM: vphadddq %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddq_epi32(a);
}

__m128i test_mm_haddw_epu8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddubw
  // CHECK-ASM: vphaddubw %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddw_epu8(a);
}

__m128i test_mm_haddd_epu8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddubd
  // CHECK-ASM: vphaddubd %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddd_epu8(a);
}

__m128i test_mm_haddq_epu8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddubq
  // CHECK-ASM: vphaddubq %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddq_epu8(a);
}

__m128i test_mm_haddd_epu16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphadduwd
  // CHECK-ASM: vphadduwd %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddd_epu16(a);
}

__m128i test_mm_haddq_epu16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphadduwq
  // CHECK-ASM: vphadduwq %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddq_epu16(a);
}

__m128i test_mm_haddq_epu32(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddudq
  // CHECK-ASM: vphaddudq %xmm{{.*}}, %xmm{{.*}}
  return _mm_haddq_epu32(a);
}

__m128i test_mm_hsubw_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphsubbw
  // CHECK-ASM: vphsubbw %xmm{{.*}}, %xmm{{.*}}
  return _mm_hsubw_epi8(a);
}

__m128i test_mm_hsubd_epi16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphsubwd
  // CHECK-ASM: vphsubwd %xmm{{.*}}, %xmm{{.*}}
  return _mm_hsubd_epi16(a);
}

__m128i test_mm_hsubq_epi32(__m128i a) {
  // CHECK: @llvm.x86.xop.vphsubdq
  // CHECK-ASM: vphsubdq %xmm{{.*}}, %xmm{{.*}}
  return _mm_hsubq_epi32(a);
}

__m128i test_mm_cmov_si128(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpcmov
  // CHECK-ASM: vpcmov %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmov_si128(a, b, c);
}

__m256i test_mm256_cmov_si256(__m256i a, __m256i b, __m256i c) {
  // CHECK: @llvm.x86.xop.vpcmov.256
  // CHECK-ASM: vpcmov %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_cmov_si256(a, b, c);
}

__m128i test_mm_perm_epi8(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpperm
  // CHECK-ASM: vpperm %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_perm_epi8(a, b, c);
}

__m128i test_mm_rot_epi8(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vprotb
  // CHECK-ASM: vprotb %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_rot_epi8(a, b);
}

__m128i test_mm_rot_epi16(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vprotw
  // CHECK-ASM: vprotw %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_rot_epi16(a, b);
}

__m128i test_mm_rot_epi32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vprotd
  // CHECK-ASM: vprotd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_rot_epi32(a, b);
}

__m128i test_mm_rot_epi64(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vprotq
  // CHECK-ASM: vprotq %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_rot_epi64(a, b);
}

__m128i test_mm_roti_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vprotbi
  // CHECK-ASM: vprotb $1, %xmm{{.*}}, %xmm{{.*}}
  return _mm_roti_epi8(a, 1);
}

__m128i test_mm_roti_epi16(__m128i a) {
  // CHECK: @llvm.x86.xop.vprotwi
  // CHECK-ASM: vprotw $50, %xmm{{.*}}, %xmm{{.*}}
  return _mm_roti_epi16(a, 50);
}

__m128i test_mm_roti_epi32(__m128i a) {
  // CHECK: @llvm.x86.xop.vprotdi
  // CHECK-ASM: vprotd $226, %xmm{{.*}}, %xmm{{.*}}
  return _mm_roti_epi32(a, -30);
}

__m128i test_mm_roti_epi64(__m128i a) {
  // CHECK: @llvm.x86.xop.vprotqi
  // CHECK-ASM: vprotq $100, %xmm{{.*}}, %xmm{{.*}}
  return _mm_roti_epi64(a, 100);
}

__m128i test_mm_shl_epi8(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpshlb
  // CHECK-ASM: vpshlb %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_shl_epi8(a, b);
}

__m128i test_mm_shl_epi16(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpshlw
  // CHECK-ASM: vpshlw %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_shl_epi16(a, b);
}

__m128i test_mm_shl_epi32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpshld
  // CHECK-ASM: vpshld %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_shl_epi32(a, b);
}

__m128i test_mm_shl_epi64(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpshlq
  // CHECK-ASM: vpshlq %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_shl_epi64(a, b);
}

__m128i test_mm_sha_epi8(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpshab
  // CHECK-ASM: vpshab %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_sha_epi8(a, b);
}

__m128i test_mm_sha_epi16(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpshaw
  // CHECK-ASM: vpshaw %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_sha_epi16(a, b);
}

__m128i test_mm_sha_epi32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpshad
  // CHECK-ASM: vpshad %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_sha_epi32(a, b);
}

__m128i test_mm_sha_epi64(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpshaq
  // CHECK-ASM: vpshaq %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_sha_epi64(a, b);
}

__m128i test_mm_com_epu8(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpcomub
  // CHECK-ASM: vpcomltub %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_com_epu8(a, b, 0);
}

__m128i test_mm_com_epu16(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpcomuw
  // CHECK-ASM: vpcomltuw %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_com_epu16(a, b, 0);
}

__m128i test_mm_com_epu32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpcomud
  // CHECK-ASM: vpcomltud %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_com_epu32(a, b, 0);
}

__m128i test_mm_com_epu64(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpcomuq
  // CHECK-ASM: vpcomltuq %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_com_epu64(a, b, 0);
}

__m128i test_mm_com_epi8(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpcomb
  // CHECK-ASM: vpcomltb %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_com_epi8(a, b, 0);
}

__m128i test_mm_com_epi16(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpcomw
  // CHECK-ASM: vpcomltw %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_com_epi16(a, b, 0);
}

__m128i test_mm_com_epi32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpcomd
  // CHECK-ASM: vpcomltd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_com_epi32(a, b, 0);
}

__m128i test_mm_com_epi64(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vpcomq
  // CHECK-ASM: vpcomltq %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_com_epi64(a, b, 0);
}

__m128d test_mm_permute2_pd(__m128d a, __m128d b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpermil2pd
  // CHECK-ASM: vpermil2pd $0, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_permute2_pd(a, b, c, 0);
}

__m256d test_mm256_permute2_pd(__m256d a, __m256d b, __m256i c) {
  // CHECK: @llvm.x86.xop.vpermil2pd.256
  // CHECK-ASM: vpermil2pd $0, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_permute2_pd(a, b, c, 0);
}

__m128 test_mm_permute2_ps(__m128 a, __m128 b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpermil2ps
  // CHECK-ASM: vpermil2ps $0, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_permute2_ps(a, b, c, 0);
}

__m256 test_mm256_permute2_ps(__m256 a, __m256 b, __m256i c) {
  // CHECK: @llvm.x86.xop.vpermil2ps.256
  // CHECK-ASM: vpermil2ps $0, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_permute2_ps(a, b, c, 0);
}

__m128 test_mm_frcz_ss(__m128 a) {
  // CHECK: @llvm.x86.xop.vfrcz.ss
  // CHECK-ASM: vfrczss %xmm{{.*}}, %xmm{{.*}}
  return _mm_frcz_ss(a);
}

__m128d test_mm_frcz_sd(__m128d a) {
  // CHECK: @llvm.x86.xop.vfrcz.sd
  // CHECK-ASM: vfrczsd %xmm{{.*}}, %xmm{{.*}}
  return _mm_frcz_sd(a);
}

__m128 test_mm_frcz_ps(__m128 a) {
  // CHECK: @llvm.x86.xop.vfrcz.ps
  // CHECK-ASM: vfrczps %xmm{{.*}}, %xmm{{.*}}
  return _mm_frcz_ps(a);
}

__m128d test_mm_frcz_pd(__m128d a) {
  // CHECK: @llvm.x86.xop.vfrcz.pd
  // CHECK-ASM: vfrczpd %xmm{{.*}}, %xmm{{.*}}
  return _mm_frcz_pd(a);
}

__m256 test_mm256_frcz_ps(__m256 a) {
  // CHECK: @llvm.x86.xop.vfrcz.ps.256
  // CHECK-ASM: vfrczps %ymm{{.*}}, %ymm{{.*}}
  return _mm256_frcz_ps(a);
}

__m256d test_mm256_frcz_pd(__m256d a) {
  // CHECK: @llvm.x86.xop.vfrcz.pd.256
  // CHECK-ASM: vfrczpd %ymm{{.*}}, %ymm{{.*}}
  return _mm256_frcz_pd(a);
}
