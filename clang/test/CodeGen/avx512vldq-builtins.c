// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512dq -target-feature +avx512vl -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m256i test_mm256_mullo_epi64 (__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mullo_epi64
  // CHECK: mul <4 x i64>
  return _mm256_mullo_epi64(__A, __B);
}

__m256i test_mm256_mask_mullo_epi64 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.256
  return (__m256i) _mm256_mask_mullo_epi64 ( __W, __U, __A, __B);
}

__m256i test_mm256_maskz_mullo_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.256
  return (__m256i) _mm256_maskz_mullo_epi64 (__U, __A, __B);
}

__m128i test_mm_mullo_epi64 (__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mullo_epi64
  // CHECK: mul <2 x i64>
  return (__m128i) _mm_mullo_epi64(__A, __B);
}

__m128i test_mm_mask_mullo_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.128
  return (__m128i) _mm_mask_mullo_epi64 ( __W, __U, __A, __B);
}

__m128i test_mm_maskz_mullo_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.128
  return (__m128i) _mm_maskz_mullo_epi64 (__U, __A, __B);
}

__m256d test_mm256_mask_andnot_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.256
  return (__m256d) _mm256_mask_andnot_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_andnot_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.256
  return (__m256d) _mm256_maskz_andnot_pd (__U, __A, __B);
}

__m128d test_mm_mask_andnot_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.128
  return (__m128d) _mm_mask_andnot_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_andnot_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.128
  return (__m128d) _mm_maskz_andnot_pd (__U, __A, __B);
}

__m256 test_mm256_mask_andnot_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.256
  return (__m256) _mm256_mask_andnot_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_andnot_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.256
  return (__m256) _mm256_maskz_andnot_ps (__U, __A, __B);
}

__m128 test_mm_mask_andnot_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.128
  return (__m128) _mm_mask_andnot_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_andnot_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.128
  return (__m128) _mm_maskz_andnot_ps (__U, __A, __B);
}

__m256d test_mm256_mask_and_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.256
  return (__m256d) _mm256_mask_and_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_and_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.256
  return (__m256d) _mm256_maskz_and_pd (__U, __A, __B);
}

__m128d test_mm_mask_and_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.128
  return (__m128d) _mm_mask_and_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_and_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.128
  return (__m128d) _mm_maskz_and_pd (__U, __A, __B);
}

__m256 test_mm256_mask_and_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.256
  return (__m256) _mm256_mask_and_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_and_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.256
  return (__m256) _mm256_maskz_and_ps (__U, __A, __B);
}

__m128 test_mm_mask_and_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.128
  return (__m128) _mm_mask_and_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_and_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.128
  return (__m128) _mm_maskz_and_ps (__U, __A, __B);
}

__m256d test_mm256_mask_xor_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.256
  return (__m256d) _mm256_mask_xor_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_xor_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.256
  return (__m256d) _mm256_maskz_xor_pd (__U, __A, __B);
}

__m128d test_mm_mask_xor_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.128
  return (__m128d) _mm_mask_xor_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_xor_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.128
  return (__m128d) _mm_maskz_xor_pd (__U, __A, __B);
}

__m256 test_mm256_mask_xor_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.256
  return (__m256) _mm256_mask_xor_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_xor_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.256
  return (__m256) _mm256_maskz_xor_ps (__U, __A, __B);
}

__m128 test_mm_mask_xor_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.128
    return (__m128) _mm_mask_xor_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_xor_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.128
  return (__m128) _mm_maskz_xor_ps (__U, __A, __B);
}

__m256d test_mm256_mask_or_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.256
  return (__m256d) _mm256_mask_or_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_or_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.256
  return (__m256d) _mm256_maskz_or_pd (__U, __A, __B);
}

__m128d test_mm_mask_or_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.128
  return (__m128d) _mm_mask_or_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_or_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.128
  return (__m128d) _mm_maskz_or_pd (__U, __A, __B);
}

__m256 test_mm256_mask_or_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.256
  return (__m256) _mm256_mask_or_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_or_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.256
  return (__m256) _mm256_maskz_or_ps (__U, __A, __B);
}

__m128 test_mm_mask_or_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.128
  return (__m128) _mm_mask_or_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_or_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.128
  return (__m128) _mm_maskz_or_ps(__U, __A, __B);
}

__m128i test_mm_cvtpd_epi64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.128
  return _mm_cvtpd_epi64(__A); 
}

__m128i test_mm_mask_cvtpd_epi64(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.128
  return _mm_mask_cvtpd_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtpd_epi64(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.128
  return _mm_maskz_cvtpd_epi64(__U, __A); 
}

__m256i test_mm256_cvtpd_epi64(__m256d __A) {
  // CHECK-LABEL: @test_mm256_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.256
  return _mm256_cvtpd_epi64(__A); 
}

__m256i test_mm256_mask_cvtpd_epi64(__m256i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.256
  return _mm256_mask_cvtpd_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtpd_epi64(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2qq.256
  return _mm256_maskz_cvtpd_epi64(__U, __A); 
}

__m128i test_mm_cvtpd_epu64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.128
  return _mm_cvtpd_epu64(__A); 
}

__m128i test_mm_mask_cvtpd_epu64(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.128
  return _mm_mask_cvtpd_epu64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtpd_epu64(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.128
  return _mm_maskz_cvtpd_epu64(__U, __A); 
}

__m256i test_mm256_cvtpd_epu64(__m256d __A) {
  // CHECK-LABEL: @test_mm256_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.256
  return _mm256_cvtpd_epu64(__A); 
}

__m256i test_mm256_mask_cvtpd_epu64(__m256i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.256
  return _mm256_mask_cvtpd_epu64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtpd_epu64(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd2uqq.256
  return _mm256_maskz_cvtpd_epu64(__U, __A); 
}

__m128i test_mm_cvtps_epi64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.128
  return _mm_cvtps_epi64(__A); 
}

__m128i test_mm_mask_cvtps_epi64(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.128
  return _mm_mask_cvtps_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtps_epi64(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.128
  return _mm_maskz_cvtps_epi64(__U, __A); 
}

__m256i test_mm256_cvtps_epi64(__m128 __A) {
  // CHECK-LABEL: @test_mm256_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.256
  return _mm256_cvtps_epi64(__A); 
}

__m256i test_mm256_mask_cvtps_epi64(__m256i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.256
  return _mm256_mask_cvtps_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtps_epi64(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps2qq.256
  return _mm256_maskz_cvtps_epi64(__U, __A); 
}

__m128i test_mm_cvtps_epu64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.128
  return _mm_cvtps_epu64(__A); 
}

__m128i test_mm_mask_cvtps_epu64(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.128
  return _mm_mask_cvtps_epu64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtps_epu64(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.128
  return _mm_maskz_cvtps_epu64(__U, __A); 
}

__m256i test_mm256_cvtps_epu64(__m128 __A) {
  // CHECK-LABEL: @test_mm256_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.256
  return _mm256_cvtps_epu64(__A); 
}

__m256i test_mm256_mask_cvtps_epu64(__m256i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.256
  return _mm256_mask_cvtps_epu64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtps_epu64(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps2uqq.256
  return _mm256_maskz_cvtps_epu64(__U, __A); 
}

__m128d test_mm_cvtepi64_pd(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2pd.128
  return _mm_cvtepi64_pd(__A); 
}

__m128d test_mm_mask_cvtepi64_pd(__m128d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2pd.128
  return _mm_mask_cvtepi64_pd(__W, __U, __A); 
}

__m128d test_mm_maskz_cvtepi64_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2pd.128
  return _mm_maskz_cvtepi64_pd(__U, __A); 
}

__m256d test_mm256_cvtepi64_pd(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2pd.256
  return _mm256_cvtepi64_pd(__A); 
}

__m256d test_mm256_mask_cvtepi64_pd(__m256d __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2pd.256
  return _mm256_mask_cvtepi64_pd(__W, __U, __A); 
}

__m256d test_mm256_maskz_cvtepi64_pd(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2pd.256
  return _mm256_maskz_cvtepi64_pd(__U, __A); 
}

__m128 test_mm_cvtepi64_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2ps.128
  return _mm_cvtepi64_ps(__A); 
}

__m128 test_mm_mask_cvtepi64_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2ps.128
  return _mm_mask_cvtepi64_ps(__W, __U, __A); 
}

__m128 test_mm_maskz_cvtepi64_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2ps.128
  return _mm_maskz_cvtepi64_ps(__U, __A); 
}

__m128 test_mm256_cvtepi64_ps(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2ps.256
  return _mm256_cvtepi64_ps(__A); 
}

__m128 test_mm256_mask_cvtepi64_ps(__m128 __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2ps.256
  return _mm256_mask_cvtepi64_ps(__W, __U, __A); 
}

__m128 test_mm256_maskz_cvtepi64_ps(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2ps.256
  return _mm256_maskz_cvtepi64_ps(__U, __A); 
}

__m128i test_mm_cvttpd_epi64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.128
  return _mm_cvttpd_epi64(__A); 
}

__m128i test_mm_mask_cvttpd_epi64(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.128
  return _mm_mask_cvttpd_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvttpd_epi64(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.128
  return _mm_maskz_cvttpd_epi64(__U, __A); 
}

__m256i test_mm256_cvttpd_epi64(__m256d __A) {
  // CHECK-LABEL: @test_mm256_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.256
  return _mm256_cvttpd_epi64(__A); 
}

__m256i test_mm256_mask_cvttpd_epi64(__m256i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.256
  return _mm256_mask_cvttpd_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvttpd_epi64(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2qq.256
  return _mm256_maskz_cvttpd_epi64(__U, __A); 
}

__m128i test_mm_cvttpd_epu64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.128
  return _mm_cvttpd_epu64(__A); 
}

__m128i test_mm_mask_cvttpd_epu64(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.128
  return _mm_mask_cvttpd_epu64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvttpd_epu64(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.128
  return _mm_maskz_cvttpd_epu64(__U, __A); 
}

__m256i test_mm256_cvttpd_epu64(__m256d __A) {
  // CHECK-LABEL: @test_mm256_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.256
  return _mm256_cvttpd_epu64(__A); 
}

__m256i test_mm256_mask_cvttpd_epu64(__m256i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.256
  return _mm256_mask_cvttpd_epu64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvttpd_epu64(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd2uqq.256
  return _mm256_maskz_cvttpd_epu64(__U, __A); 
}

__m128i test_mm_cvttps_epi64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.128
  return _mm_cvttps_epi64(__A); 
}

__m128i test_mm_mask_cvttps_epi64(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.128
  return _mm_mask_cvttps_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvttps_epi64(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.128
  return _mm_maskz_cvttps_epi64(__U, __A); 
}

__m256i test_mm256_cvttps_epi64(__m128 __A) {
  // CHECK-LABEL: @test_mm256_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.256
  return _mm256_cvttps_epi64(__A); 
}

__m256i test_mm256_mask_cvttps_epi64(__m256i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.256
  return _mm256_mask_cvttps_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvttps_epi64(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps2qq.256
  return _mm256_maskz_cvttps_epi64(__U, __A); 
}

__m128i test_mm_cvttps_epu64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.128
  return _mm_cvttps_epu64(__A); 
}

__m128i test_mm_mask_cvttps_epu64(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.128
  return _mm_mask_cvttps_epu64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvttps_epu64(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.128
  return _mm_maskz_cvttps_epu64(__U, __A); 
}

__m256i test_mm256_cvttps_epu64(__m128 __A) {
  // CHECK-LABEL: @test_mm256_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.256
  return _mm256_cvttps_epu64(__A); 
}

__m256i test_mm256_mask_cvttps_epu64(__m256i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.256
  return _mm256_mask_cvttps_epu64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvttps_epu64(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps2uqq.256
  return _mm256_maskz_cvttps_epu64(__U, __A); 
}

__m128d test_mm_cvtepu64_pd(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2pd.128
  return _mm_cvtepu64_pd(__A); 
}

__m128d test_mm_mask_cvtepu64_pd(__m128d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2pd.128
  return _mm_mask_cvtepu64_pd(__W, __U, __A); 
}

__m128d test_mm_maskz_cvtepu64_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2pd.128
  return _mm_maskz_cvtepu64_pd(__U, __A); 
}

__m256d test_mm256_cvtepu64_pd(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2pd.256
  return _mm256_cvtepu64_pd(__A); 
}

__m256d test_mm256_mask_cvtepu64_pd(__m256d __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2pd.256
  return _mm256_mask_cvtepu64_pd(__W, __U, __A); 
}

__m256d test_mm256_maskz_cvtepu64_pd(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2pd.256
  return _mm256_maskz_cvtepu64_pd(__U, __A); 
}

__m128 test_mm_cvtepu64_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2ps.128
  return _mm_cvtepu64_ps(__A); 
}

__m128 test_mm_mask_cvtepu64_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2ps.128
  return _mm_mask_cvtepu64_ps(__W, __U, __A); 
}

__m128 test_mm_maskz_cvtepu64_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2ps.128
  return _mm_maskz_cvtepu64_ps(__U, __A); 
}

__m128 test_mm256_cvtepu64_ps(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2ps.256
  return _mm256_cvtepu64_ps(__A); 
}

__m128 test_mm256_mask_cvtepu64_ps(__m128 __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2ps.256
  return _mm256_mask_cvtepu64_ps(__W, __U, __A); 
}

__m128 test_mm256_maskz_cvtepu64_ps(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2ps.256
  return _mm256_maskz_cvtepu64_ps(__U, __A); 
}

__m128d test_mm_range_pd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.128
  return _mm_range_pd(__A, __B, 4); 
}

__m128d test_mm_mask_range_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.128
  return _mm_mask_range_pd(__W, __U, __A, __B, 4); 
}

__m128d test_mm_maskz_range_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.128
  return _mm_maskz_range_pd(__U, __A, __B, 4); 
}

__m256d test_mm256_range_pd(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.256
  return _mm256_range_pd(__A, __B, 4); 
}

__m256d test_mm256_mask_range_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.256
  return _mm256_mask_range_pd(__W, __U, __A, __B, 4); 
}

__m256d test_mm256_maskz_range_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.256
  return _mm256_maskz_range_pd(__U, __A, __B, 4); 
}

__m128 test_mm_range_ps(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.128
  return _mm_range_ps(__A, __B, 4); 
}

__m128 test_mm_mask_range_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.128
  return _mm_mask_range_ps(__W, __U, __A, __B, 4); 
}

__m128 test_mm_maskz_range_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.128
  return _mm_maskz_range_ps(__U, __A, __B, 4); 
}

__m256 test_mm256_range_ps(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.256
  return _mm256_range_ps(__A, __B, 4); 
}

__m256 test_mm256_mask_range_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.256
  return _mm256_mask_range_ps(__W, __U, __A, __B, 4); 
}

__m256 test_mm256_maskz_range_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.256
  return _mm256_maskz_range_ps(__U, __A, __B, 4); 
}

__m128d test_mm_reduce_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.128
  return _mm_reduce_pd(__A, 4); 
}

__m128d test_mm_mask_reduce_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.128
  return _mm_mask_reduce_pd(__W, __U, __A, 4); 
}

__m128d test_mm_maskz_reduce_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.128
  return _mm_maskz_reduce_pd(__U, __A, 4); 
}

__m256d test_mm256_reduce_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.256
  return _mm256_reduce_pd(__A, 4); 
}

__m256d test_mm256_mask_reduce_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.256
  return _mm256_mask_reduce_pd(__W, __U, __A, 4); 
}

__m256d test_mm256_maskz_reduce_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.256
  return _mm256_maskz_reduce_pd(__U, __A, 4); 
}

__m128 test_mm_reduce_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.128
  return _mm_reduce_ps(__A, 4); 
}

__m128 test_mm_mask_reduce_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.128
  return _mm_mask_reduce_ps(__W, __U, __A, 4); 
}

__m128 test_mm_maskz_reduce_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.128
  return _mm_maskz_reduce_ps(__U, __A, 4); 
}

__m256 test_mm256_reduce_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.256
  return _mm256_reduce_ps(__A, 4); 
}

__m256 test_mm256_mask_reduce_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.256
  return _mm256_mask_reduce_ps(__W, __U, __A, 4); 
}

__m256 test_mm256_maskz_reduce_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.256
  return _mm256_maskz_reduce_ps(__U, __A, 4); 
}

__mmask8 test_mm_movepi32_mask(__m128i __A) {
  // CHECK-LABEL: @test_mm_movepi32_mask
  // CHECK: @llvm.x86.avx512.cvtd2mask.128
  return _mm_movepi32_mask(__A); 
}

__mmask8 test_mm256_movepi32_mask(__m256i __A) {
  // CHECK-LABEL: @test_mm256_movepi32_mask
  // CHECK: @llvm.x86.avx512.cvtd2mask.256
  return _mm256_movepi32_mask(__A); 
}

__m128i test_mm_movm_epi32(__mmask8 __A) {
  // CHECK-LABEL: @test_mm_movm_epi32
  // CHECK: @llvm.x86.avx512.cvtmask2d.128
  return _mm_movm_epi32(__A); 
}

__m256i test_mm256_movm_epi32(__mmask8 __A) {
  // CHECK-LABEL: @test_mm256_movm_epi32
  // CHECK: @llvm.x86.avx512.cvtmask2d.256
  return _mm256_movm_epi32(__A); 
}

__m128i test_mm_movm_epi64(__mmask8 __A) {
  // CHECK-LABEL: @test_mm_movm_epi64
  // CHECK: @llvm.x86.avx512.cvtmask2q.128
  return _mm_movm_epi64(__A); 
}

__m256i test_mm256_movm_epi64(__mmask8 __A) {
  // CHECK-LABEL: @test_mm256_movm_epi64
  // CHECK: @llvm.x86.avx512.cvtmask2q.256
  return _mm256_movm_epi64(__A); 
}

__mmask8 test_mm_movepi64_mask(__m128i __A) {
  // CHECK-LABEL: @test_mm_movepi64_mask
  // CHECK: @llvm.x86.avx512.cvtq2mask.128
  return _mm_movepi64_mask(__A); 
}

__mmask8 test_mm256_movepi64_mask(__m256i __A) {
  // CHECK-LABEL: @test_mm256_movepi64_mask
  // CHECK: @llvm.x86.avx512.cvtq2mask.256
  return _mm256_movepi64_mask(__A); 
}


__m256 test_mm256_broadcast_f32x2(__m128 __A) {
  // CHECK-LABEL: @test_mm256_broadcast_f32x2
  // CHECK: @llvm.x86.avx512.mask.broadcastf32x2
  return _mm256_broadcast_f32x2(__A); 
}

__m256 test_mm256_mask_broadcast_f32x2(__m256 __O, __mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_f32x2
  // CHECK: @llvm.x86.avx512.mask.broadcastf32x2
  return _mm256_mask_broadcast_f32x2(__O, __M, __A); 
}

__m256 test_mm256_maskz_broadcast_f32x2(__mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_f32x2
  // CHECK: @llvm.x86.avx512.mask.broadcastf32x2
  return _mm256_maskz_broadcast_f32x2(__M, __A); 
}

__m256d test_mm256_broadcast_f64x2(__m128d __A) {
  // CHECK-LABEL: @test_mm256_broadcast_f64x2
  // CHECK: @llvm.x86.avx512.mask.broadcastf64x2
  return _mm256_broadcast_f64x2(__A); 
}

__m256d test_mm256_mask_broadcast_f64x2(__m256d __O, __mmask8 __M, __m128d __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_f64x2
  // CHECK: @llvm.x86.avx512.mask.broadcastf64x2
  return _mm256_mask_broadcast_f64x2(__O, __M, __A); 
}

__m256d test_mm256_maskz_broadcast_f64x2(__mmask8 __M, __m128d __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_f64x2
  // CHECK: @llvm.x86.avx512.mask.broadcastf64x2
  return _mm256_maskz_broadcast_f64x2(__M, __A); 
}

__m128i test_mm_broadcast_i32x2(__m128i __A) {
  // CHECK-LABEL: @test_mm_broadcast_i32x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x2
  return _mm_broadcast_i32x2(__A); 
}

__m128i test_mm_mask_broadcast_i32x2(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_broadcast_i32x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x2
  return _mm_mask_broadcast_i32x2(__O, __M, __A); 
}

__m128i test_mm_maskz_broadcast_i32x2(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcast_i32x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x2
  return _mm_maskz_broadcast_i32x2(__M, __A); 
}

__m256i test_mm256_broadcast_i32x2(__m128i __A) {
  // CHECK-LABEL: @test_mm256_broadcast_i32x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x2
  return _mm256_broadcast_i32x2(__A); 
}

__m256i test_mm256_mask_broadcast_i32x2(__m256i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_i32x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x2
  return _mm256_mask_broadcast_i32x2(__O, __M, __A); 
}

__m256i test_mm256_maskz_broadcast_i32x2(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_i32x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x2
  return _mm256_maskz_broadcast_i32x2(__M, __A); 
}

__m256i test_mm256_broadcast_i64x2(__m128i __A) {
  // CHECK-LABEL: @test_mm256_broadcast_i64x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti64x2
  return _mm256_broadcast_i64x2(__A); 
}

__m256i test_mm256_mask_broadcast_i64x2(__m256i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_i64x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti64x2
  return _mm256_mask_broadcast_i64x2(__O, __M, __A); 
}

__m256i test_mm256_maskz_broadcast_i64x2(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_i64x2
  // CHECK: @llvm.x86.avx512.mask.broadcasti64x2
  return _mm256_maskz_broadcast_i64x2(__M, __A); 
}

__m128d test_mm256_extractf64x2_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_extractf64x2_pd
  // CHECK: @llvm.x86.avx512.mask.vextractf64x2
  return _mm256_extractf64x2_pd(__A, 1); 
}

__m128d test_mm256_mask_extractf64x2_pd(__m128d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_extractf64x2_pd
  // CHECK: @llvm.x86.avx512.mask.vextractf64x2
  return _mm256_mask_extractf64x2_pd(__W, __U, __A, 1); 
}

__m128d test_mm256_maskz_extractf64x2_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_extractf64x2_pd
  // CHECK: @llvm.x86.avx512.mask.vextractf64x2
  return _mm256_maskz_extractf64x2_pd(__U, __A, 1); 
}

__m128i test_mm256_extracti64x2_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_extracti64x2_epi64
  // CHECK: @llvm.x86.avx512.mask.vextracti64x2
  return _mm256_extracti64x2_epi64(__A, 1); 
}

__m128i test_mm256_mask_extracti64x2_epi64(__m128i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_extracti64x2_epi64
  // CHECK: @llvm.x86.avx512.mask.vextracti64x2
  return _mm256_mask_extracti64x2_epi64(__W, __U, __A, 1); 
}

__m128i test_mm256_maskz_extracti64x2_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_extracti64x2_epi64
  // CHECK: @llvm.x86.avx512.mask.vextracti64x2
  return _mm256_maskz_extracti64x2_epi64(__U, __A, 1); 
}

__m256d test_mm256_insertf64x2(__m256d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm256_insertf64x2
  // CHECK: @llvm.x86.avx512.mask.insertf64x2
  return _mm256_insertf64x2(__A, __B, 1); 
}

__m256d test_mm256_mask_insertf64x2(__m256d __W, __mmask8 __U, __m256d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm256_mask_insertf64x2
  // CHECK: @llvm.x86.avx512.mask.insertf64x2
  return _mm256_mask_insertf64x2(__W, __U, __A, __B, 1); 
}

__m256d test_mm256_maskz_insertf64x2(__mmask8 __U, __m256d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm256_maskz_insertf64x2
  // CHECK: @llvm.x86.avx512.mask.insertf64x2
  return _mm256_maskz_insertf64x2(__U, __A, __B, 1); 
}

__m256i test_mm256_inserti64x2(__m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_inserti64x2
  // CHECK: @llvm.x86.avx512.mask.inserti64x2
  return _mm256_inserti64x2(__A, __B, 1); 
}

__m256i test_mm256_mask_inserti64x2(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_inserti64x2
  // CHECK: @llvm.x86.avx512.mask.inserti64x2
  return _mm256_mask_inserti64x2(__W, __U, __A, __B, 1); 
}

__m256i test_mm256_maskz_inserti64x2(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_inserti64x2
  // CHECK: @llvm.x86.avx512.mask.inserti64x2
  return _mm256_maskz_inserti64x2(__U, __A, __B, 1); 
}

__mmask8 test_mm_mask_fpclass_pd_mask(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_fpclass_pd_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.pd.128
  return _mm_mask_fpclass_pd_mask(__U, __A, 2); 
}

__mmask8 test_mm_fpclass_pd_mask(__m128d __A) {
  // CHECK-LABEL: @test_mm_fpclass_pd_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.pd.128
  return _mm_fpclass_pd_mask(__A, 2); 
}

__mmask8 test_mm256_mask_fpclass_pd_mask(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_fpclass_pd_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.pd.256
  return _mm256_mask_fpclass_pd_mask(__U, __A, 2); 
}

__mmask8 test_mm256_fpclass_pd_mask(__m256d __A) {
  // CHECK-LABEL: @test_mm256_fpclass_pd_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.pd.256
  return _mm256_fpclass_pd_mask(__A, 2); 
}

__mmask8 test_mm_mask_fpclass_ps_mask(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_fpclass_ps_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.ps.128
  return _mm_mask_fpclass_ps_mask(__U, __A, 2); 
}

__mmask8 test_mm_fpclass_ps_mask(__m128 __A) {
  // CHECK-LABEL: @test_mm_fpclass_ps_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.ps.128
  return _mm_fpclass_ps_mask(__A, 2); 
}

__mmask8 test_mm256_mask_fpclass_ps_mask(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_fpclass_ps_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.ps.256
  return _mm256_mask_fpclass_ps_mask(__U, __A, 2); 
}

__mmask8 test_mm256_fpclass_ps_mask(__m256 __A) {
  // CHECK-LABEL: @test_mm256_fpclass_ps_mask
  // CHECK: @llvm.x86.avx512.mask.fpclass.ps.256
  return _mm256_fpclass_ps_mask(__A, 2); 
}
