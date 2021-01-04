// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -target-feature +avx512vl -ffp-exception-behavior=strict -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=CONSTRAINED %s
// RUN: %clang_cc1 -fexperimental-new-pass-manager -flax-vector-conversions=none -fms-compatibility -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +avx512f -target-feature +avx512vl -ffp-exception-behavior=strict -emit-llvm -o - -Wall -Werror | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=CONSTRAINED %s

#include <immintrin.h>

__m128 test_mm_mask_cvtph_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // COMMON-LABEL: @test_mm_mask_cvtph_ps
  // COMMONIR: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // COMMONIR: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // COMMONIR: bitcast <4 x i16> %{{.*}} to <4 x half>
  // UNCONSTRAINED: fpext <4 x half> %{{.*}} to <4 x float>
  // CONSTRAINED: call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %{{.*}}, metadata !"fpexcept.strict")
  // COMMONIR: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_cvtph_ps(__W, __U, __A);
}

__m128 test_mm_maskz_cvtph_ps(__mmask8 __U, __m128i __A) {
  // COMMON-LABEL: @test_mm_maskz_cvtph_ps
  // COMMONIR: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // COMMONIR: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // COMMONIR: bitcast <4 x i16> %{{.*}} to <4 x half>
  // UNCONSTRAINED: fpext <4 x half> %{{.*}} to <4 x float>
  // CONSTRAINED: call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %{{.*}}, metadata !"fpexcept.strict")
  // COMMONIR: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_cvtph_ps(__U, __A);
}

__m256 test_mm256_mask_cvtph_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  // COMMON-LABEL: @test_mm256_mask_cvtph_ps
  // COMMONIR: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // COMMONIR: bitcast <8 x i16> %{{.*}} to <8 x half>
  // UNCONSTRAINED: fpext <8 x half> %{{.*}} to <8 x float>
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.fpext.v8f32.v8f16(<8 x half> %{{.*}}, metadata !"fpexcept.strict") 
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_cvtph_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_cvtph_ps(__mmask8 __U, __m128i __A) {
  // COMMON-LABEL: @test_mm256_maskz_cvtph_ps
  // COMMONIR: bitcast <2 x i64> %{{.*}} to <8 x i16>
  // COMMONIR: bitcast <8 x i16> %{{.*}} to <8 x half>
  // UNCONSTRAINED: fpext <8 x half> %{{.*}} to <8 x float>
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.fpext.v8f32.v8f16(<8 x half> %{{.*}}, metadata !"fpexcept.strict") 
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_cvtph_ps(__U, __A);
}

__m128i test_mm_mask_cvtps_ph(__m128i __W, __mmask8 __U, __m128 __A) {
  // COMMON-LABEL: @test_mm_mask_cvtps_ph
  // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_mask_cvtps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_maskz_cvtps_ph(__mmask8 __U, __m128 __A) {
  // COMMON-LABEL: @test_mm_maskz_cvtps_ph
  // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_maskz_cvtps_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_mask_cvtps_ph(__m128i __W, __mmask8 __U, __m256 __A) {
  // COMMON-LABEL: @test_mm256_mask_cvtps_ph
  // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_mask_cvtps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm256_maskz_cvtps_ph(__mmask8 __U, __m256 __A) {
  // COMMON-LABEL: @test_mm256_maskz_cvtps_ph
  // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_maskz_cvtps_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_mask_cvt_roundps_ph(__m128i __W, __mmask8 __U, __m128 __A) {
  // COMMON-LABEL: @test_mm_mask_cvt_roundps_ph
  // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO);
}

__m128i test_mm_maskz_cvt_roundps_ph(__mmask8 __U, __m128 __A) {
  // COMMON-LABEL: @test_mm_maskz_cvt_roundps_ph
  // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_TO_ZERO);
}

__m128i test_mm256_mask_cvt_roundps_ph(__m128i __W, __mmask8 __U, __m256 __A) {
  // COMMON-LABEL: @test_mm256_mask_cvt_roundps_ph
  // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_TO_ZERO);
}

__m128i test_mm256_maskz_cvt_roundps_ph(__mmask8 __U, __m256 __A) {
  // COMMON-LABEL: @test_mm256_maskz_cvt_roundps_ph
  // COMMONIR: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_TO_ZERO);
}
