// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2-256 \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2-256 \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128bh test_mm_minmaxne_pbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_minmaxne_pbh(
  // CHECK: call <8 x bfloat> @llvm.x86.avx10.vminmaxnepbf16128(
  return _mm_minmaxne_pbh(__A, __B, 127);
}

__m128bh test_mm_mask_minmaxne_pbh(__m128bh __A, __mmask8 __B, __m128bh __C, __m128bh __D) {
  // CHECK-LABEL: @test_mm_mask_minmaxne_pbh(
  // CHECK: call <8 x bfloat> @llvm.x86.avx10.vminmaxnepbf16128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask_minmaxne_pbh(__A, __B, __C, __D, 127);
}

__m128bh test_mm_maskz_minmaxne_pbh(__mmask8 __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_maskz_minmaxne_pbh(
  // CHECK: call <8 x bfloat> @llvm.x86.avx10.vminmaxnepbf16128(
  // CHECK: zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_minmaxne_pbh(__A, __B, __C, 127);
}

__m256bh test_mm256_minmaxne_pbh(__m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_minmaxne_pbh(
  // CHECK: call <16 x bfloat> @llvm.x86.avx10.vminmaxnepbf16256(
  return _mm256_minmaxne_pbh(__A, __B, 127);
}

__m256bh test_mm256_mask_minmaxne_pbh(__m256bh __A, __mmask16 __B, __m256bh __C, __m256bh __D) {
  // CHECK-LABEL: @test_mm256_mask_minmaxne_pbh(
  // CHECK: call <16 x bfloat> @llvm.x86.avx10.vminmaxnepbf16256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask_minmaxne_pbh(__A, __B, __C, __D, 127);
}

__m256bh test_mm256_maskz_minmaxne_pbh(__mmask16 __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_maskz_minmaxne_pbh(
  // CHECK: call <16 x bfloat> @llvm.x86.avx10.vminmaxnepbf16256(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_minmaxne_pbh(__A, __B, __C, 127);
}

__m128d test_mm_minmax_pd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_minmax_pd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxpd128(
  return _mm_minmax_pd(__A, __B, 127);
}

__m128d test_mm_mask_minmax_pd(__m128d __A, __mmask8 __B, __m128d __C, __m128d __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_pd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxpd128(
  return _mm_mask_minmax_pd(__A, __B, __C, __D, 127);
}

__m128d test_mm_maskz_minmax_pd(__mmask8 __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_pd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxpd128(
  return _mm_maskz_minmax_pd(__A, __B, __C, 127);
}

__m256d test_mm256_minmax_pd(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_minmax_pd(
  // CHECK: call <4 x double> @llvm.x86.avx10.mask.vminmaxpd256.round(
  return _mm256_minmax_pd(__A, __B, 127);
}

__m256d test_mm256_mask_minmax_pd(__m256d __A, __mmask8 __B, __m256d __C, __m256d __D) {
  // CHECK-LABEL: @test_mm256_mask_minmax_pd(
  // CHECK: call <4 x double> @llvm.x86.avx10.mask.vminmaxpd256.round(
  return _mm256_mask_minmax_pd(__A, __B, __C, __D, 127);
}

__m256d test_mm256_maskz_minmax_pd(__mmask8 __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_minmax_pd(
  // CHECK: call <4 x double> @llvm.x86.avx10.mask.vminmaxpd256.round(
  return _mm256_maskz_minmax_pd(__A, __B, __C, 127);
}

__m256d test_mm256_minmax_round_pd(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_minmax_round_pd(
  // CHECK: call <4 x double> @llvm.x86.avx10.mask.vminmaxpd256.round(
  return _mm256_minmax_round_pd(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_mask_minmax_round_pd(__m256d __A, __mmask8 __B, __m256d __C, __m256d __D) {
  // CHECK-LABEL: @test_mm256_mask_minmax_round_pd(
  // CHECK: call <4 x double> @llvm.x86.avx10.mask.vminmaxpd256.round(
  return _mm256_mask_minmax_round_pd(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m256d test_mm256_maskz_minmax_round_pd(__mmask8 __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_minmax_round_pd(
  // CHECK: call <4 x double> @llvm.x86.avx10.mask.vminmaxpd256.round(
  return _mm256_maskz_minmax_round_pd(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}

__m128h test_mm_minmax_ph(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_minmax_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxph128(
  return _mm_minmax_ph(__A, __B, 127);
}

__m128h test_mm_mask_minmax_ph(__m128h __A, __mmask8 __B, __m128h __C, __m128h __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxph128(
  return _mm_mask_minmax_ph(__A, __B, __C, __D, 127);
}

__m128h test_mm_maskz_minmax_ph(__mmask8 __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxph128(
  return _mm_maskz_minmax_ph(__A, __B, __C, 127);
}

__m256h test_mm256_minmax_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_minmax_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vminmaxph256.round(
  return _mm256_minmax_ph(__A, __B, 127);
}

__m256h test_mm256_mask_minmax_ph(__m256h __A, __mmask16 __B, __m256h __C, __m256h __D) {
  // CHECK-LABEL: @test_mm256_mask_minmax_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vminmaxph256.round(
  return _mm256_mask_minmax_ph(__A, __B, __C, __D, 127);
}

__m256h test_mm256_maskz_minmax_ph(__mmask16 __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_minmax_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vminmaxph256.round(
  return _mm256_maskz_minmax_ph(__A, __B, __C, 127);
}

__m256h test_mm256_minmax_round_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_minmax_round_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vminmaxph256.round(
  return _mm256_minmax_round_ph(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_minmax_round_ph(__m256h __A, __mmask16 __B, __m256h __C, __m256h __D) {
  // CHECK-LABEL: @test_mm256_mask_minmax_round_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vminmaxph256.round(
  return _mm256_mask_minmax_round_ph(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_minmax_round_ph(__mmask16 __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_minmax_round_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vminmaxph256.round(
  return _mm256_maskz_minmax_round_ph(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}

__m128 test_mm_minmax_ps(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_minmax_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxps128(
  return _mm_minmax_ps(__A, __B, 127);
}

__m128 test_mm_mask_minmax_ps(__m128 __A, __mmask8 __B, __m128 __C, __m128 __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxps128(
  return _mm_mask_minmax_ps(__A, __B, __C, __D, 127);
}

__m128 test_mm_maskz_minmax_ps(__mmask8 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxps128(
  return _mm_maskz_minmax_ps(__A, __B, __C, 127);
}

__m256 test_mm256_minmax_ps(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_minmax_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vminmaxps256.round(
  return _mm256_minmax_ps(__A, __B, 127);
}

__m256 test_mm256_mask_minmax_ps(__m256 __A, __mmask8 __B, __m256 __C, __m256 __D) {
  // CHECK-LABEL: @test_mm256_mask_minmax_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vminmaxps256.round(
  return _mm256_mask_minmax_ps(__A, __B, __C, __D, 127);
}

__m256 test_mm256_maskz_minmax_ps(__mmask8 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_minmax_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vminmaxps256.round(
  return _mm256_maskz_minmax_ps(__A, __B, __C, 127);
}

__m256 test_mm256_minmax_round_ps(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_minmax_round_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vminmaxps256.round(
  return _mm256_minmax_round_ps(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_mask_minmax_round_ps(__m256 __A, __mmask8 __B, __m256 __C, __m256 __D) {
  // CHECK-LABEL: @test_mm256_mask_minmax_round_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vminmaxps256.round(
  return _mm256_mask_minmax_round_ps(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m256 test_mm256_maskz_minmax_round_ps(__mmask8 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_minmax_round_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vminmaxps256.round(
  return _mm256_maskz_minmax_round_ps(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}

__m128d test_mm_minmax_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_minmax_sd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxsd.round(
  return _mm_minmax_sd(__A, __B, 127);
}

__m128d test_mm_mask_minmax_sd(__m128d __A, __mmask8 __B, __m128d __C, __m128d __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_sd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxsd.round(
  return _mm_mask_minmax_sd(__A, __B, __C, __D, 127);
}

__m128d test_mm_maskz_minmax_sd(__mmask8 __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_sd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxsd.round(
  return _mm_maskz_minmax_sd(__A, __B, __C, 127);
}

__m128d test_mm_minmax_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_minmax_round_sd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxsd.round(
  return _mm_minmax_round_sd(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_minmax_round_sd(__m128d __A, __mmask8 __B, __m128d __C, __m128d __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_round_sd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxsd.round(
  return _mm_mask_minmax_round_sd(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_minmax_round_sd(__mmask8 __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_round_sd(
  // CHECK: call <2 x double> @llvm.x86.avx10.mask.vminmaxsd.round(
  return _mm_maskz_minmax_round_sd(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}

__m128h test_mm_minmax_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_minmax_sh(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxsh.round(
  return _mm_minmax_sh(__A, __B, 127);
}

__m128h test_mm_mask_minmax_sh(__m128h __A, __mmask8 __B, __m128h __C, __m128h __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_sh(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxsh.round(
  return _mm_mask_minmax_sh(__A, __B, __C, __D, 127);
}

__m128h test_mm_maskz_minmax_sh(__mmask8 __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_sh(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxsh.round(
  return _mm_maskz_minmax_sh(__A, __B, __C, 127);
}

__m128h test_mm_minmax_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_minmax_round_sh(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxsh.round(
  return _mm_minmax_round_sh(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_minmax_round_sh(__m128h __A, __mmask8 __B, __m128h __C, __m128h __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_round_sh(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxsh.round(
  return _mm_mask_minmax_round_sh(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_minmax_round_sh(__mmask8 __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_round_sh(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vminmaxsh.round(
  return _mm_maskz_minmax_round_sh(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}

__m128 test_mm_minmax_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_minmax_ss(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxss.round(
  return _mm_minmax_ss(__A, __B, 127);
}

__m128 test_mm_mask_minmax_ss(__m128 __A, __mmask8 __B, __m128 __C, __m128 __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_ss(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxss.round(
  return _mm_mask_minmax_ss(__A, __B, __C, __D, 127);
}

__m128 test_mm_maskz_minmax_ss(__mmask8 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_ss(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxss.round(
  return _mm_maskz_minmax_ss(__A, __B, __C, 127);
}

__m128 test_mm_minmax_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_minmax_round_ss(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxss.round(
  return _mm_minmax_round_ss(__A, __B, 127, _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_minmax_round_ss(__m128 __A, __mmask8 __B, __m128 __C, __m128 __D) {
  // CHECK-LABEL: @test_mm_mask_minmax_round_ss(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxss.round(
  return _mm_mask_minmax_round_ss(__A, __B, __C, __D, 127, _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_minmax_round_ss(__mmask8 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_minmax_round_ss(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vminmaxss.round(
  return _mm_maskz_minmax_round_ss(__A, __B, __C, 127, _MM_FROUND_NO_EXC);
}
