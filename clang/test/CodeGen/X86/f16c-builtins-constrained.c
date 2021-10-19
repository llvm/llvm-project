// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +f16c -emit-llvm -ffp-exception-behavior=strict -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

float test_cvtsh_ss(unsigned short a) {
  // CHECK-LABEL: test_cvtsh_ss
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 3
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 4
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 5
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 6
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 7
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: bitcast <4 x i16> %{{.*}} to <4 x half>
  // CHECK: call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %{{.*}}, metadata !"fpexcept.strict")
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  return _cvtsh_ss(a);
}

unsigned short test_cvtss_sh(float a) {
  // CHECK-LABEL: test_cvtss_sh
  // CHECK: insertelement <4 x float> undef, float %{{.*}}, i32 0
  // CHECK: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 1
  // CHECK: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 2
  // CHECK: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 3
  // CHECK: call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %{{.*}}, i32 0)
  // CHECK: extractelement <8 x i16> %{{.*}}, i32 0
  return _cvtss_sh(a, 0);
}

__m128 test_mm_cvtph_ps(__m128i a) {
  // CHECK-LABEL: test_mm_cvtph_ps
  // CHECK: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: bitcast <4 x i16> %{{.*}} to <4 x half>
  // CHECK: call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %{{.*}}, metadata !"fpexcept.strict")
  return _mm_cvtph_ps(a);
}

__m256 test_mm256_cvtph_ps(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtph_ps
  // CHECK: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK: bitcast <8 x i16> %{{.*}} to <8 x half>
  // CHECK: call <8 x float> @llvm.experimental.constrained.fpext.v8f32.v8f16(<8 x half> %{{.*}}, metadata !"fpexcept.strict")
  return _mm256_cvtph_ps(a);
}

__m128i test_mm_cvtps_ph(__m128 a) {
  // CHECK-LABEL: test_mm_cvtps_ph
  // CHECK: call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %{{.*}}, i32 0)
  return _mm_cvtps_ph(a, 0);
}

__m128i test_mm256_cvtps_ph(__m256 a) {
  // CHECK-LABEL: test_mm256_cvtps_ph
  // CHECK: call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %{{.*}}, i32 0)
  return _mm256_cvtps_ph(a, 0);
}
