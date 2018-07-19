// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

__m256i test_mm256_mullo_epi64 (__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mullo_epi64
  // CHECK: mul <4 x i64>
  return _mm256_mullo_epi64(__A, __B);
}

__m256i test_mm256_mask_mullo_epi64 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_mullo_epi64
  // CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return (__m256i) _mm256_mask_mullo_epi64 ( __W, __U, __A, __B);
}

__m256i test_mm256_maskz_mullo_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_mullo_epi64
  // CHECK: mul <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return (__m256i) _mm256_maskz_mullo_epi64 (__U, __A, __B);
}

__m128i test_mm_mullo_epi64 (__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mullo_epi64
  // CHECK: mul <2 x i64>
  return (__m128i) _mm_mullo_epi64(__A, __B);
}

__m128i test_mm_mask_mullo_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_mullo_epi64
  // CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return (__m128i) _mm_mask_mullo_epi64 ( __W, __U, __A, __B);
}

__m128i test_mm_maskz_mullo_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_mullo_epi64
  // CHECK: mul <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return (__m128i) _mm_maskz_mullo_epi64 (__U, __A, __B);
}

__m256d test_mm256_mask_andnot_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_andnot_pd
  // CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return (__m256d) _mm256_mask_andnot_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_andnot_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_andnot_pd
  // CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return (__m256d) _mm256_maskz_andnot_pd (__U, __A, __B);
}

__m128d test_mm_mask_andnot_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_andnot_pd
  // CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return (__m128d) _mm_mask_andnot_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_andnot_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_andnot_pd
  // CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return (__m128d) _mm_maskz_andnot_pd (__U, __A, __B);
}

__m256 test_mm256_mask_andnot_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_andnot_ps
  // CHECK: xor <8 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  // CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return (__m256) _mm256_mask_andnot_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_andnot_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_andnot_ps
  // CHECK: xor <8 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  // CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return (__m256) _mm256_maskz_andnot_ps (__U, __A, __B);
}

__m128 test_mm_mask_andnot_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_andnot_ps
  // CHECK: xor <4 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1>
  // CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return (__m128) _mm_mask_andnot_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_andnot_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_andnot_ps
  // CHECK: xor <4 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1>
  // CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return (__m128) _mm_maskz_andnot_ps (__U, __A, __B);
}

__m256d test_mm256_mask_and_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_and_pd
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return (__m256d) _mm256_mask_and_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_and_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_and_pd
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return (__m256d) _mm256_maskz_and_pd (__U, __A, __B);
}

__m128d test_mm_mask_and_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_and_pd
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return (__m128d) _mm_mask_and_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_and_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_and_pd
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return (__m128d) _mm_maskz_and_pd (__U, __A, __B);
}

__m256 test_mm256_mask_and_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_and_ps
  // CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return (__m256) _mm256_mask_and_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_and_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_and_ps
  // CHECK: and <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return (__m256) _mm256_maskz_and_ps (__U, __A, __B);
}

__m128 test_mm_mask_and_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_and_ps
  // CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return (__m128) _mm_mask_and_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_and_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_and_ps
  // CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return (__m128) _mm_maskz_and_ps (__U, __A, __B);
}

__m256d test_mm256_mask_xor_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_xor_pd
  // CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return (__m256d) _mm256_mask_xor_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_xor_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_xor_pd
  // CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return (__m256d) _mm256_maskz_xor_pd (__U, __A, __B);
}

__m128d test_mm_mask_xor_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_xor_pd
  // CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return (__m128d) _mm_mask_xor_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_xor_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_xor_pd
  // CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return (__m128d) _mm_maskz_xor_pd (__U, __A, __B);
}

__m256 test_mm256_mask_xor_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_xor_ps
  // CHECK: xor <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return (__m256) _mm256_mask_xor_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_xor_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_xor_ps
  // CHECK: xor <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return (__m256) _mm256_maskz_xor_ps (__U, __A, __B);
}

__m128 test_mm_mask_xor_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_xor_ps
  // CHECK: xor <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return (__m128) _mm_mask_xor_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_xor_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_xor_ps
  // CHECK: xor <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return (__m128) _mm_maskz_xor_ps (__U, __A, __B);
}

__m256d test_mm256_mask_or_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_or_pd
  // CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return (__m256d) _mm256_mask_or_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_or_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_or_pd
  // CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return (__m256d) _mm256_maskz_or_pd (__U, __A, __B);
}

__m128d test_mm_mask_or_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_or_pd
  // CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return (__m128d) _mm_mask_or_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_or_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_or_pd
  // CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return (__m128d) _mm_maskz_or_pd (__U, __A, __B);
}

__m256 test_mm256_mask_or_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_or_ps
  // CHECK: or <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return (__m256) _mm256_mask_or_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_or_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_or_ps
  // CHECK: or <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return (__m256) _mm256_maskz_or_ps (__U, __A, __B);
}

__m128 test_mm_mask_or_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_or_ps
  // CHECK: or <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return (__m128) _mm_mask_or_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_or_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_or_ps
  // CHECK: or <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
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
  // CHECK: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: %extract.i = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: %vpmovm2.i = sext <4 x i1> %extract.i to <4 x i32>
  return _mm_movm_epi32(__A); 
}

__m256i test_mm256_movm_epi32(__mmask8 __A) {
  // CHECK-LABEL: @test_mm256_movm_epi32
  // CHECK: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: %vpmovm2.i = sext <8 x i1> %{{.*}} to <8 x i32>
  return _mm256_movm_epi32(__A); 
}

__m128i test_mm_movm_epi64(__mmask8 __A) {
  // CHECK-LABEL: @test_mm_movm_epi64
  // CHECK: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: %extract.i = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: %vpmovm2.i = sext <2 x i1> %extract.i to <2 x i64>
  return _mm_movm_epi64(__A); 
}

__m256i test_mm256_movm_epi64(__mmask8 __A) {
  // CHECK-LABEL: @test_mm256_movm_epi64
  // CHECK: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: %extract.i = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: %vpmovm2.i = sext <4 x i1> %extract.i to <4 x i64>
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
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  return _mm256_broadcast_f32x2(__A); 
}

__m256 test_mm256_mask_broadcast_f32x2(__m256 __O, __mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_f32x2
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_broadcast_f32x2(__O, __M, __A); 
}

__m256 test_mm256_maskz_broadcast_f32x2(__mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_f32x2
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_broadcast_f32x2(__M, __A); 
}

__m256d test_mm256_broadcast_f64x2(double const* __A) {
  // CHECK-LABEL: @test_mm256_broadcast_f64x2
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  return _mm256_broadcast_f64x2(_mm_loadu_pd(__A)); 
}

__m256d test_mm256_mask_broadcast_f64x2(__m256d __O, __mmask8 __M, double const* __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_f64x2
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_broadcast_f64x2(__O, __M, _mm_loadu_pd(__A)); 
}

__m256d test_mm256_maskz_broadcast_f64x2(__mmask8 __M, double const* __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_f64x2
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_broadcast_f64x2(__M, _mm_loadu_pd(__A)); 
}

__m128i test_mm_broadcast_i32x2(__m128i __A) {
  // CHECK-LABEL: @test_mm_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  return _mm_broadcast_i32x2(__A); 
}

__m128i test_mm_mask_broadcast_i32x2(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_broadcast_i32x2(__O, __M, __A); 
}

__m128i test_mm_maskz_broadcast_i32x2(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_broadcast_i32x2(__M, __A); 
}

__m256i test_mm256_broadcast_i32x2(__m128i __A) {
  // CHECK-LABEL: @test_mm256_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  return _mm256_broadcast_i32x2(__A); 
}

__m256i test_mm256_mask_broadcast_i32x2(__m256i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_broadcast_i32x2(__O, __M, __A); 
}

__m256i test_mm256_maskz_broadcast_i32x2(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_i32x2
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_broadcast_i32x2(__M, __A); 
}

__m256i test_mm256_broadcast_i64x2(__m128i const* __A) {
  // CHECK-LABEL: @test_mm256_broadcast_i64x2
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  return _mm256_broadcast_i64x2(_mm_loadu_si128(__A)); 
}

__m256i test_mm256_mask_broadcast_i64x2(__m256i __O, __mmask8 __M, __m128i const* __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_i64x2
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_broadcast_i64x2(__O, __M, _mm_loadu_si128(__A)); 
}

__m256i test_mm256_maskz_broadcast_i64x2(__mmask8 __M, __m128i const* __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_i64x2
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_broadcast_i64x2(__M, _mm_loadu_si128(__A)); 
}

__m128d test_mm256_extractf64x2_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_extractf64x2_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> zeroinitializer, <2 x i32> <i32 2, i32 3>
  return _mm256_extractf64x2_pd(__A, 1); 
}

__m128d test_mm256_mask_extractf64x2_pd(__m128d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_extractf64x2_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> zeroinitializer, <2 x i32> <i32 2, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm256_mask_extractf64x2_pd(__W, __U, __A, 1); 
}

__m128d test_mm256_maskz_extractf64x2_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_extractf64x2_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> zeroinitializer, <2 x i32> <i32 2, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm256_maskz_extractf64x2_pd(__U, __A, 1); 
}

__m128i test_mm256_extracti64x2_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_extracti64x2_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> zeroinitializer, <2 x i32> <i32 2, i32 3>
  return _mm256_extracti64x2_epi64(__A, 1); 
}

__m128i test_mm256_mask_extracti64x2_epi64(__m128i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_extracti64x2_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> zeroinitializer, <2 x i32> <i32 2, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm256_mask_extracti64x2_epi64(__W, __U, __A, 1); 
}

__m128i test_mm256_maskz_extracti64x2_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_extracti64x2_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> zeroinitializer, <2 x i32> <i32 2, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm256_maskz_extracti64x2_epi64(__U, __A, 1); 
}

__m256d test_mm256_insertf64x2(__m256d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm256_insertf64x2
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_insertf64x2(__A, __B, 1); 
}

__m256d test_mm256_mask_insertf64x2(__m256d __W, __mmask8 __U, __m256d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm256_mask_insertf64x2
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_insertf64x2(__W, __U, __A, __B, 1); 
}

__m256d test_mm256_maskz_insertf64x2(__mmask8 __U, __m256d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm256_maskz_insertf64x2
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_insertf64x2(__U, __A, __B, 1); 
}

__m256i test_mm256_inserti64x2(__m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_inserti64x2
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_inserti64x2(__A, __B, 1); 
}

__m256i test_mm256_mask_inserti64x2(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_inserti64x2
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_inserti64x2(__W, __U, __A, __B, 1); 
}

__m256i test_mm256_maskz_inserti64x2(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_inserti64x2
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
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
