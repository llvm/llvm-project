// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=UNCONSTRAINED --check-prefix=COMMON --check-prefix=COMMONIR
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -ffp-exception-behavior=maytrap -DSTRICT=1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CONSTRAINED --check-prefix=COMMON --check-prefix=COMMONIR
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -ffp-exception-behavior=maytrap -DSTRICT=1 -S -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-ASM --check-prefix=COMMON

// FIXME: Every instance of "fpexcept.maytrap" is wrong.
#ifdef STRICT
// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)
#endif


#include <immintrin.h>

__m512d test_mm512_cvtepi64_pd(__m512i __A) {
  // COMMON-LABEL: test_mm512_cvtepi64_pd
  // UNCONSTRAINED: sitofp <8 x i64> %{{.*}} to <8 x double>
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.sitofp.v8f64.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
  // CHECK-ASM: vcvtqq2pd
  return _mm512_cvtepi64_pd(__A);
}

__m512d test_mm512_mask_cvtepi64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_mask_cvtepi64_pd
  // UNCONSTRAINED: sitofp <8 x i64> %{{.*}} to <8 x double>
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.sitofp.v8f64.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  // CHECK-ASM: vcvtqq2pd
  return _mm512_mask_cvtepi64_pd(__W, __U, __A);
}

__m512d test_mm512_maskz_cvtepi64_pd(__mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_maskz_cvtepi64_pd
  // UNCONSTRAINED: sitofp <8 x i64> %{{.*}} to <8 x double>
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.sitofp.v8f64.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  // CHECK-ASM: vcvtqq2pd
  return _mm512_maskz_cvtepi64_pd(__U, __A);
}

__m512d test_mm512_cvt_roundepi64_pd(__m512i __A) {
  // COMMON-LABEL: test_mm512_cvt_roundepi64_pd
  // COMMONIR: @llvm.x86.avx512.sitofp.round.v8f64.v8i64
  // CHECK-ASM: vcvtqq2pd
  return _mm512_cvt_roundepi64_pd(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_cvt_roundepi64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_mask_cvt_roundepi64_pd
  // COMMONIR: @llvm.x86.avx512.sitofp.round.v8f64.v8i64
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  // CHECK-ASM: vcvtqq2pd
  return _mm512_mask_cvt_roundepi64_pd(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_cvt_roundepi64_pd(__mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_maskz_cvt_roundepi64_pd
  // COMMONIR: @llvm.x86.avx512.sitofp.round.v8f64.v8i64
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  // CHECK-ASM: vcvtqq2pd
  return _mm512_maskz_cvt_roundepi64_pd(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_cvtepi64_ps(__m512i __A) {
  // COMMON-LABEL: test_mm512_cvtepi64_ps
  // UNCONSTRAINED: sitofp <8 x i64> %{{.*}} to <8 x float>
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.sitofp.v8f32.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK-ASM: vcvtqq2ps
  return _mm512_cvtepi64_ps(__A);
}

__m256 test_mm512_mask_cvtepi64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_mask_cvtepi64_ps
  // UNCONSTRAINED: sitofp <8 x i64> %{{.*}} to <8 x float>
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.sitofp.v8f32.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK-ASM: vcvtqq2ps
  return _mm512_mask_cvtepi64_ps(__W, __U, __A);
}

__m256 test_mm512_maskz_cvtepi64_ps(__mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_maskz_cvtepi64_ps
  // UNCONSTRAINED: sitofp <8 x i64> %{{.*}} to <8 x float>
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.sitofp.v8f32.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK-ASM: vcvtqq2ps
  return _mm512_maskz_cvtepi64_ps(__U, __A);
}

__m256 test_mm512_cvt_roundepi64_ps(__m512i __A) {
  // COMMON-LABEL: test_mm512_cvt_roundepi64_ps
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // COMMONIR: @llvm.x86.avx512.sitofp.round.v8f32.v8i64
  // CHECK-ASM: vcvtqq2ps
  return _mm512_cvt_roundepi64_ps(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_mask_cvt_roundepi64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_mask_cvt_roundepi64_ps
  // COMMONIR: @llvm.x86.avx512.sitofp.round.v8f32.v8i64
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK-ASM: vcvtqq2ps
  return _mm512_mask_cvt_roundepi64_ps(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_maskz_cvt_roundepi64_ps(__mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_maskz_cvt_roundepi64_ps
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // COMMONIR: @llvm.x86.avx512.sitofp.round.v8f32.v8i64
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK-ASM: vcvtqq2ps
  return _mm512_maskz_cvt_roundepi64_ps(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_cvtepu64_pd(__m512i __A) {
  // COMMON-LABEL: test_mm512_cvtepu64_pd
  // UNCONSTRAINED: uitofp <8 x i64> %{{.*}} to <8 x double>
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.uitofp.v8f64.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
  // CHECK-ASM: vcvtuqq2pd
  return _mm512_cvtepu64_pd(__A);
}

__m512d test_mm512_mask_cvtepu64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_mask_cvtepu64_pd
  // UNCONSTRAINED: uitofp <8 x i64> %{{.*}} to <8 x double>
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.uitofp.v8f64.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  // CHECK-ASM: vcvtuqq2pd
  return _mm512_mask_cvtepu64_pd(__W, __U, __A);
}

__m512d test_mm512_maskz_cvtepu64_pd(__mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_maskz_cvtepu64_pd
  // UNCONSTRAINED: uitofp <8 x i64> %{{.*}} to <8 x double>
  // CONSTRAINED: call <8 x double> @llvm.experimental.constrained.uitofp.v8f64.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  // CHECK-ASM: vcvtuqq2pd
  return _mm512_maskz_cvtepu64_pd(__U, __A);
}

__m512d test_mm512_cvt_roundepu64_pd(__m512i __A) {
  // COMMON-LABEL: test_mm512_cvt_roundepu64_pd
  // COMMONIR: @llvm.x86.avx512.uitofp.round.v8f64.v8i64
  // CHECK-ASM: vcvtuqq2pd
  return _mm512_cvt_roundepu64_pd(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_cvt_roundepu64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_mask_cvt_roundepu64_pd
  // COMMONIR: @llvm.x86.avx512.uitofp.round.v8f64.v8i64
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  // CHECK-ASM: vcvtuqq2pd
  return _mm512_mask_cvt_roundepu64_pd(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_cvt_roundepu64_pd(__mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_maskz_cvt_roundepu64_pd
  // COMMONIR: @llvm.x86.avx512.uitofp.round.v8f64.v8i64
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
  // CHECK-ASM: vcvtuqq2pd
  return _mm512_maskz_cvt_roundepu64_pd(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_cvtepu64_ps(__m512i __A) {
  // COMMON-LABEL: test_mm512_cvtepu64_ps
  // UNCONSTRAINED: uitofp <8 x i64> %{{.*}} to <8 x float>
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.uitofp.v8f32.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CHECK-ASM: vcvtuqq2ps
  return _mm512_cvtepu64_ps(__A);
}

__m256 test_mm512_mask_cvtepu64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_mask_cvtepu64_ps
  // UNCONSTRAINED: uitofp <8 x i64> %{{.*}} to <8 x float>
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.uitofp.v8f32.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK-ASM: vcvtuqq2ps
  return _mm512_mask_cvtepu64_ps(__W, __U, __A);
}

__m256 test_mm512_maskz_cvtepu64_ps(__mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_maskz_cvtepu64_ps
  // UNCONSTRAINED: uitofp <8 x i64> %{{.*}} to <8 x float>
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.uitofp.v8f32.v8i64(<8 x i64> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK-ASM: vcvtuqq2ps
  return _mm512_maskz_cvtepu64_ps(__U, __A);
}

__m256 test_mm512_cvt_roundepu64_ps(__m512i __A) {
  // COMMON-LABEL: test_mm512_cvt_roundepu64_ps
  // COMMONIR: @llvm.x86.avx512.uitofp.round.v8f32.v8i64
  // CHECK-ASM: vcvtuqq2ps
  return _mm512_cvt_roundepu64_ps(__A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_mask_cvt_roundepu64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_mask_cvt_roundepu64_ps
  // COMMONIR: @llvm.x86.avx512.uitofp.round.v8f32.v8i64
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK-ASM: vcvtuqq2ps
  return _mm512_mask_cvt_roundepu64_ps(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256 test_mm512_maskz_cvt_roundepu64_ps(__mmask8 __U, __m512i __A) {
  // COMMON-LABEL: test_mm512_maskz_cvt_roundepu64_ps
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // CONSTRAINED: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 0, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // COMMONIR: @llvm.x86.avx512.uitofp.round.v8f32.v8i64
  // COMMONIR: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK-ASM: vcvtuqq2ps
  return _mm512_maskz_cvt_roundepu64_ps(__U, __A, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

