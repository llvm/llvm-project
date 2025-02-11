// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2-256 \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2-256 \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128h test_mm_cvtx2ps_ph(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvtx2ps_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.128
  return _mm_cvtx2ps_ph(__A, __B);
}

__m128h test_mm_mask_cvtx2ps_ph(__m128h __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvtx2ps_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.128
  return _mm_mask_cvtx2ps_ph(__W, __U, __A, __B);
}

__m128h test_mm_maskz_cvtx2ps_ph(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtx2ps_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.128
  return _mm_maskz_cvtx2ps_ph(__U, __A, __B);
}

__m256h test_mm256_cvtx2ps_ph(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvtx2ps_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.256
  return _mm256_cvtx2ps_ph(__A, __B);
}

__m256h test_mm256_mask_cvtx2ps_ph(__m256h __W, __mmask16 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtx2ps_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.256
  return _mm256_mask_cvtx2ps_ph(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_cvtx2ps_ph(__mmask16 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtx2ps_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.256
  return _mm256_maskz_cvtx2ps_ph(__U, __A, __B);
}

__m256h test_mm256_cvtx_round2ps_ph(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvtx_round2ps_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.256(
  return _mm256_cvtx_round2ps_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_mask_cvtx_round2ps_ph(__m256h __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtx_round2ps_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.256(
  return _mm256_mask_cvtx_round2ps_ph(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm256_maskz_cvtx_round2ps_ph(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtx_round2ps_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.256(
  return _mm256_maskz_cvtx_round2ps_ph(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128i test_mm_cvtbiasph_bf8(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtbiasph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8128(
  return _mm_cvtbiasph_bf8(__A, __B);
}

__m128i test_mm_mask_cvtbiasph_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiasph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8128(
  return _mm_mask_cvtbiasph_bf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiasph_bf8(__mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiasph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8128(
  return _mm_maskz_cvtbiasph_bf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiasph_bf8(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtbiasph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8256(
  return _mm256_cvtbiasph_bf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiasph_bf8(__m128i __W, __mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiasph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8256(
  return _mm256_mask_cvtbiasph_bf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiasph_bf8(__mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiasph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8256(
  return _mm256_maskz_cvtbiasph_bf8(__U, __A, __B);
}

__m128i test_mm_cvtbiassph_bf8(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtbiassph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s128(
  return _mm_cvtbiassph_bf8(__A, __B);
}

__m128i test_mm_mask_cvtbiassph_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiassph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s128(
  return _mm_mask_cvtbiassph_bf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiassph_bf8(__mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiassph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s128(
  return _mm_maskz_cvtbiassph_bf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiassph_bf8(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtbiassph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s256(
  return _mm256_cvtbiassph_bf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiassph_bf8(__m128i __W, __mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiassph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s256(
  return _mm256_mask_cvtbiassph_bf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiassph_bf8(__mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiassph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s256(
  return _mm256_maskz_cvtbiassph_bf8(__U, __A, __B);
}

__m128i test_mm_cvtbiasph_hf8(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtbiasph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8128(
  return _mm_cvtbiasph_hf8(__A, __B);
}

__m128i test_mm_mask_cvtbiasph_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiasph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8128(
  return _mm_mask_cvtbiasph_hf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiasph_hf8(__mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiasph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8128(
  return _mm_maskz_cvtbiasph_hf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiasph_hf8(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtbiasph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8256(
  return _mm256_cvtbiasph_hf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiasph_hf8(__m128i __W, __mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiasph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8256(
  return _mm256_mask_cvtbiasph_hf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiasph_hf8(__mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiasph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8256(
  return _mm256_maskz_cvtbiasph_hf8(__U, __A, __B);
}

__m128i test_mm_cvtbiassph_hf8(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtbiassph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s128(
  return _mm_cvtbiassph_hf8(__A, __B);
}

__m128i test_mm_mask_cvtbiassph_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiassph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s128(
  return _mm_mask_cvtbiassph_hf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiassph_hf8(__mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiassph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s128(
  return _mm_maskz_cvtbiassph_hf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiassph_hf8(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtbiassph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s256(
  return _mm256_cvtbiassph_hf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiassph_hf8(__m128i __W, __mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiassph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s256(
  return _mm256_mask_cvtbiassph_hf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiassph_hf8(__mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiassph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s256(
  return _mm256_maskz_cvtbiassph_hf8(__U, __A, __B);
}

__m128i test_mm_cvt2ph_bf8(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvt2ph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2bf8128(
  return _mm_cvt2ph_bf8(__A, __B);
}

__m128i test_mm_mask_cvt2ph_bf8(__m128i __W, __mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvt2ph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2bf8128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  // CHECK: ret <2 x i64> %{{.*}}
  return _mm_mask_cvt2ph_bf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvt2ph_bf8(__mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvt2ph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2bf8128(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_cvt2ph_bf8(__U, __A, __B);
}

__m256i test_mm256_cvt2ph_bf8(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvt2ph_bf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2bf8256(
  return _mm256_cvt2ph_bf8(__A, __B);
}

__m256i test_mm256_mask_cvt2ph_bf8(__m256i __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvt2ph_bf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2bf8256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  // CHECK: ret <4 x i64> %{{.*}}
  return _mm256_mask_cvt2ph_bf8(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_cvt2ph_bf8(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvt2ph_bf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2bf8256(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_cvt2ph_bf8(__U, __A, __B);
}

__m128i test_mm_cvts2ph_bf8(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvts2ph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2bf8s128(
  return _mm_cvts2ph_bf8(__A, __B);
}

__m128i test_mm_mask_cvts2ph_bf8(__m128i __W, __mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvts2ph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2bf8s128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  // CHECK: ret <2 x i64> %{{.*}}
  return _mm_mask_cvts2ph_bf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvts2ph_bf8(__mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvts2ph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2bf8s128(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_cvts2ph_bf8(__U, __A, __B);
}

__m256i test_mm256_cvts2ph_bf8(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvts2ph_bf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2bf8s256(
  return _mm256_cvts2ph_bf8(__A, __B);
}

__m256i test_mm256_mask_cvts2ph_bf8(__m256i __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvts2ph_bf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2bf8s256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  // CHECK: ret <4 x i64> %{{.*}}
  return _mm256_mask_cvts2ph_bf8(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_cvts2ph_bf8(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvts2ph_bf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2bf8s256(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_cvts2ph_bf8(__U, __A, __B);
}

__m128i test_mm_cvt2ph_hf8(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvt2ph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2hf8128(
  return _mm_cvt2ph_hf8(__A, __B);
}

__m128i test_mm_mask_cvt2ph_hf8(__m128i __W, __mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvt2ph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2hf8128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  // CHECK: ret <2 x i64> %{{.*}}
  return _mm_mask_cvt2ph_hf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvt2ph_hf8(__mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvt2ph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2hf8128(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_cvt2ph_hf8(__U, __A, __B);
}

__m256i test_mm256_cvt2ph_hf8(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvt2ph_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2hf8256(
  return _mm256_cvt2ph_hf8(__A, __B);
}

__m256i test_mm256_mask_cvt2ph_hf8(__m256i __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvt2ph_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2hf8256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  // CHECK: ret <4 x i64> %{{.*}}
  return _mm256_mask_cvt2ph_hf8(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_cvt2ph_hf8(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvt2ph_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2hf8256(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_cvt2ph_hf8(__U, __A, __B);
}

__m128i test_mm_cvts2ph_hf8(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvts2ph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2hf8s128(
  return _mm_cvts2ph_hf8(__A, __B);
}

__m128i test_mm_mask_cvts2ph_hf8(__m128i __W, __mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvts2ph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2hf8s128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  // CHECK: ret <2 x i64> %{{.*}}
  return _mm_mask_cvts2ph_hf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvts2ph_hf8(__mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvts2ph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvt2ph2hf8s128(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_cvts2ph_hf8(__U, __A, __B);
}

__m256i test_mm256_cvts2ph_hf8(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvts2ph_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2hf8s256(
  return _mm256_cvts2ph_hf8(__A, __B);
}

__m256i test_mm256_mask_cvts2ph_hf8(__m256i __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvts2ph_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2hf8s256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  // CHECK: ret <4 x i64> %{{.*}}
  return _mm256_mask_cvts2ph_hf8(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_cvts2ph_hf8(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvts2ph_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvt2ph2hf8s256(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_cvts2ph_hf8(__U, __A, __B);
}

__m128h test_mm_cvthf8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvthf8(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvthf82ph128(
  return _mm_cvthf8(__A);
}

__m128h test_mm_mask_cvthf8(__m128h __A, __mmask8 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_cvthf8(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvthf82ph128(
  return _mm_mask_cvthf8(__A, __B, __C);
}

__m128h test_mm_maskz_cvthf8(__mmask8 __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_cvthf8(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvthf82ph128(
  return _mm_maskz_cvthf8(__A, __B);
}

__m256h test_mm256_cvthf8(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvthf8(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvthf82ph256(
  return _mm256_cvthf8(__A);
}

__m256h test_mm256_mask_cvthf8(__m256h __A, __mmask16 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm256_mask_cvthf8(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvthf82ph256(
  return _mm256_mask_cvthf8(__A, __B, __C);
}

__m256h test_mm256_maskz_cvthf8(__mmask16 __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvthf8(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvthf82ph256(
  return _mm256_maskz_cvthf8(__A, __B);
}

__m128i test_mm_cvtph_bf8(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8128(
  return _mm_cvtph_bf8(__A);
}

__m128i test_mm_mask_cvtph_bf8(__m128i __A, __mmask8 __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_cvtph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8128(
  return _mm_mask_cvtph_bf8(__A, __B, __C);
}

__m128i test_mm_maskz_cvtph_bf8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8128(
  return _mm_maskz_cvtph_bf8(__A, __B);
}

__m128i test_mm256_cvtph_bf8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8256(
  return _mm256_cvtph_bf8(__A);
}

__m128i test_mm256_mask_cvtph_bf8(__m128i __A, __mmask16 __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8256(
  return _mm256_mask_cvtph_bf8(__A, __B, __C);
}

__m128i test_mm256_maskz_cvtph_bf8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8256(
  return _mm256_maskz_cvtph_bf8(__A, __B);
}

__m128i test_mm_cvtsph_bf8(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtsph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8s128(
  return _mm_cvtsph_bf8(__A);
}

__m128i test_mm_mask_cvtsph_bf8(__m128i __A, __mmask8 __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_cvtsph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8s128(
  return _mm_mask_cvtsph_bf8(__A, __B, __C);
}

__m128i test_mm_maskz_cvtsph_bf8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtsph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8s128(
  return _mm_maskz_cvtsph_bf8(__A, __B);
}

__m128i test_mm256_cvtsph_bf8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtsph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8s256(
  return _mm256_cvtsph_bf8(__A);
}

__m128i test_mm256_mask_cvtsph_bf8(__m128i __A, __mmask16 __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtsph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8s256(
  return _mm256_mask_cvtsph_bf8(__A, __B, __C);
}

__m128i test_mm256_maskz_cvtsph_bf8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsph_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2bf8s256(
  return _mm256_maskz_cvtsph_bf8(__A, __B);
}

__m128i test_mm_cvtph_hf8(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8128(
  return _mm_cvtph_hf8(__A);
}

__m128i test_mm_mask_cvtph_hf8(__m128i __A, __mmask8 __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_cvtph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8128(
  return _mm_mask_cvtph_hf8(__A, __B, __C);
}

__m128i test_mm_maskz_cvtph_hf8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8128(
  return _mm_maskz_cvtph_hf8(__A, __B);
}

__m128i test_mm256_cvtph_hf8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8256(
  return _mm256_cvtph_hf8(__A);
}

__m128i test_mm256_mask_cvtph_hf8(__m128i __A, __mmask16 __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8256(
  return _mm256_mask_cvtph_hf8(__A, __B, __C);
}

__m128i test_mm256_maskz_cvtph_hf8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8256(
  return _mm256_maskz_cvtph_hf8(__A, __B);
}

__m128i test_mm_cvtsph_hf8(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtsph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8s128(
  return _mm_cvtsph_hf8(__A);
}

__m128i test_mm_mask_cvtsph_hf8(__m128i __A, __mmask8 __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_cvtsph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8s128(
  return _mm_mask_cvtsph_hf8(__A, __B, __C);
}

__m128i test_mm_maskz_cvtsph_hf8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtsph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8s128(
  return _mm_maskz_cvtsph_hf8(__A, __B);
}

__m128i test_mm256_cvtsph_hf8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtsph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8s256(
  return _mm256_cvtsph_hf8(__A);
}

__m128i test_mm256_mask_cvtsph_hf8(__m128i __A, __mmask16 __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtsph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8s256(
  return _mm256_mask_cvtsph_hf8(__A, __B, __C);
}

__m128i test_mm256_maskz_cvtsph_hf8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsph_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtph2hf8s256(
  return _mm256_maskz_cvtsph_hf8(__A, __B);
}

__m256h test_mm256_cvtbf8_ph(__m128i A) {
  // CHECK-LABEL: @test_mm256_cvtbf8_ph
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: ret <16 x half> %{{.*}}
  return _mm256_cvtbf8_ph(A);
}

__m256h test_mm256_mask_cvtbf8_ph(__m256h S, __mmask16 M, __m128i A) {
  // CHECK-LABEL: @test_mm256_mask_cvtbf8_ph
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  // CHECK: ret <16 x half> %{{.*}}
  return _mm256_mask_cvtbf8_ph(S, M, A);
}

__m256h test_mm256_maskz_cvtbf8_ph(__mmask16 M, __m128i A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbf8_ph
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: ret <16 x half> %{{.*}}
  return _mm256_maskz_cvtbf8_ph(M, A);
}

__m128h test_mm_cvtbf8_ph(__m128i A) {
  // CHECK-LABEL: @test_mm_cvtbf8_ph
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: ret <8 x half> %{{.*}}
  return _mm_cvtbf8_ph(A);
}

__m128h test_mm_mask_cvtbf8_ph(__m128h S, __mmask8 M, __m128i A) {
  // CHECK-LABEL: @test_mm_mask_cvtbf8_ph
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  // CHECK: ret <8 x half> %{{.*}}
  return _mm_mask_cvtbf8_ph(S, M, A);
}

__m128h test_mm_maskz_cvtbf8_ph(__mmask8 M, __m128i A) {
  // CHECK-LABEL: @test_mm_maskz_cvtbf8_ph
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: ret <8 x half> %{{.*}}
  return _mm_maskz_cvtbf8_ph(M, A);
}
