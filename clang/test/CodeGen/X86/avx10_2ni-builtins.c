// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +avx10.2-256 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i686-unknown-unknown -target-feature +avx10.2-256 -emit-llvm -o - | FileCheck %s

#include <immintrin.h>

// VMPSADBW
__m128i test_mm_mpsadbw_epu8(__m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mpsadbw_epu8
// CHECK: @llvm.x86.sse41.mpsadbw
  return _mm_mpsadbw_epu8(__A, __B, 170);
}

__m128i test_mm_mask_mpsadbw_epu8(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_mask_mpsadbw_epu8
// CHECK: @llvm.x86.sse41.mpsadbw
// CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mpsadbw_epu8(__W, __U, __A, __B, 170);
}

__m128i test_mm_maskz_mpsadbw_epu8(__mmask8 __U, __m128i __A, __m128i __B) {
// CHECK-LABEL: @test_mm_maskz_mpsadbw_epu8
// CHECK: @llvm.x86.sse41.mpsadbw
// CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mpsadbw_epu8(__U, __A, __B, 170);
}

__m256i test_mm256_mpsadbw_epu8(__m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mpsadbw_epu8
// CHECK: @llvm.x86.avx2.mpsadbw
  return _mm256_mpsadbw_epu8(__A, __B, 170);
}

__m256i test_mm256_mask_mpsadbw_epu8(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_mask_mpsadbw_epu8
// CHECK: @llvm.x86.avx2.mpsadbw
// CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mpsadbw_epu8(__W, __U, __A, __B, 170);
}

__m256i test_mm256_maskz_mpsadbw_epu8(__mmask16 __U, __m256i __A, __m256i __B) {
// CHECK-LABEL: @test_mm256_maskz_mpsadbw_epu8
// CHECK: @llvm.x86.avx2.mpsadbw
// CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mpsadbw_epu8(__U, __A, __B, 170);
}

// YMM Rounding
__m256d test_mm256_add_round_pd(__m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_add_round_pd
// CHECK: @llvm.x86.avx10.vaddpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 11)
  return _mm256_add_round_pd(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_add_round_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_mask_add_round_pd
// CHECK: @llvm.x86.avx10.vaddpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 10)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_add_round_pd(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_add_round_pd(__mmask8 __U, __m256d __A, __m256d __B) {
// CHECK-LABEL: @test_mm256_maskz_add_round_pd
// CHECK: @llvm.x86.avx10.vaddpd256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, i32 9)
// CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_add_round_pd(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_add_round_ph(__m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_add_round_ph
// CHECK: @llvm.x86.avx10.vaddph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 11)
  return _mm256_add_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_add_round_ph(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_mask_add_round_ph
// CHECK: @llvm.x86.avx10.vaddph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 10)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_add_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_add_round_ph(__mmask8 __U, __m256h __A, __m256h __B) {
// CHECK-LABEL: @test_mm256_maskz_add_round_ph
// CHECK: @llvm.x86.avx10.vaddph256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 9)
// CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_add_round_ph(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_add_round_ps(__m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_add_round_ps
// CHECK: @llvm.x86.avx10.vaddps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 11)
  return _mm256_add_round_ps(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_add_round_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_mask_add_round_ps
// CHECK: @llvm.x86.avx10.vaddps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 10)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_add_round_ps(__W, __U, __A, __B, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_add_round_ps(__mmask8 __U, __m256 __A, __m256 __B) {
// CHECK-LABEL: @test_mm256_maskz_add_round_ps
// CHECK: @llvm.x86.avx10.vaddps256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, i32 9)
// CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_add_round_ps(__U, __A, __B, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}
