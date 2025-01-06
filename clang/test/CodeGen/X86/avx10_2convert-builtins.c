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

__m128i test_mm_cvtbiasph_pbf8(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtbiasph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8128(
  return _mm_cvtbiasph_pbf8(__A, __B);
}

__m128i test_mm_mask_cvtbiasph_pbf8(__m128i __W, __mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiasph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8128(
  return _mm_mask_cvtbiasph_pbf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiasph_pbf8(__mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiasph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8128(
  return _mm_maskz_cvtbiasph_pbf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiasph_pbf8(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtbiasph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8256(
  return _mm256_cvtbiasph_pbf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiasph_pbf8(__m128i __W, __mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiasph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8256(
  return _mm256_mask_cvtbiasph_pbf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiasph_pbf8(__mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiasph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8256(
  return _mm256_maskz_cvtbiasph_pbf8(__U, __A, __B);
}

__m128i test_mm_cvtbiassph_pbf8(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtbiassph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s128(
  return _mm_cvtbiassph_pbf8(__A, __B);
}

__m128i test_mm_mask_cvtbiassph_pbf8(__m128i __W, __mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiassph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s128(
  return _mm_mask_cvtbiassph_pbf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiassph_pbf8(__mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiassph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s128(
  return _mm_maskz_cvtbiassph_pbf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiassph_pbf8(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtbiassph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s256(
  return _mm256_cvtbiassph_pbf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiassph_pbf8(__m128i __W, __mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiassph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s256(
  return _mm256_mask_cvtbiassph_pbf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiassph_pbf8(__mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiassph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s256(
  return _mm256_maskz_cvtbiassph_pbf8(__U, __A, __B);
}

__m128i test_mm_cvtbiasph_phf8(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtbiasph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8128(
  return _mm_cvtbiasph_phf8(__A, __B);
}

__m128i test_mm_mask_cvtbiasph_phf8(__m128i __W, __mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiasph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8128(
  return _mm_mask_cvtbiasph_phf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiasph_phf8(__mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiasph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8128(
  return _mm_maskz_cvtbiasph_phf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiasph_phf8(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtbiasph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8256(
  return _mm256_cvtbiasph_phf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiasph_phf8(__m128i __W, __mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiasph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8256(
  return _mm256_mask_cvtbiasph_phf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiasph_phf8(__mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiasph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8256(
  return _mm256_maskz_cvtbiasph_phf8(__U, __A, __B);
}

__m128i test_mm_cvtbiassph_phf8(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtbiassph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s128(
  return _mm_cvtbiassph_phf8(__A, __B);
}

__m128i test_mm_mask_cvtbiassph_phf8(__m128i __W, __mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiassph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s128(
  return _mm_mask_cvtbiassph_phf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiassph_phf8(__mmask8 __U, __m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiassph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s128(
  return _mm_maskz_cvtbiassph_phf8(__U, __A, __B);
}

__m128i test_mm256_cvtbiassph_phf8(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtbiassph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s256(
  return _mm256_cvtbiassph_phf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiassph_phf8(__m128i __W, __mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiassph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s256(
  return _mm256_mask_cvtbiassph_phf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiassph_phf8(__mmask16 __U, __m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiassph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s256(
  return _mm256_maskz_cvtbiassph_phf8(__U, __A, __B);
}

__m128i test_mm_cvtne2ph_pbf8(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtne2ph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2bf8128(
  return _mm_cvtne2ph_pbf8(__A, __B);
}

__m128i test_mm_mask_cvtne2ph_pbf8(__m128i __W, __mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtne2ph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2bf8128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  // CHECK: ret <2 x i64> %{{.*}}
  return _mm_mask_cvtne2ph_pbf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtne2ph_pbf8(__mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtne2ph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2bf8128(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_cvtne2ph_pbf8(__U, __A, __B);
}

__m256i test_mm256_cvtne2ph_pbf8(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtne2ph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2bf8256(
  return _mm256_cvtne2ph_pbf8(__A, __B);
}

__m256i test_mm256_mask_cvtne2ph_pbf8(__m256i __W, __mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtne2ph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2bf8256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  // CHECK: ret <4 x i64> %{{.*}}
  return _mm256_mask_cvtne2ph_pbf8(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_cvtne2ph_pbf8(__mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtne2ph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2bf8256(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_cvtne2ph_pbf8(__U, __A, __B);
}

__m128i test_mm_cvtnes2ph_pbf8(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtnes2ph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s128(
  return _mm_cvtnes2ph_pbf8(__A, __B);
}

__m128i test_mm_mask_cvtnes2ph_pbf8(__m128i __W, __mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtnes2ph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  // CHECK: ret <2 x i64> %{{.*}}
  return _mm_mask_cvtnes2ph_pbf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtnes2ph_pbf8(__mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtnes2ph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s128(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_cvtnes2ph_pbf8(__U, __A, __B);
}

__m256i test_mm256_cvtnes2ph_pbf8(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtnes2ph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s256(
  return _mm256_cvtnes2ph_pbf8(__A, __B);
}

__m256i test_mm256_mask_cvtnes2ph_pbf8(__m256i __W, __mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtnes2ph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  // CHECK: ret <4 x i64> %{{.*}}
  return _mm256_mask_cvtnes2ph_pbf8(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_cvtnes2ph_pbf8(__mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtnes2ph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s256(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_cvtnes2ph_pbf8(__U, __A, __B);
}

__m128i test_mm_cvtne2ph_phf8(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtne2ph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2hf8128(
  return _mm_cvtne2ph_phf8(__A, __B);
}

__m128i test_mm_mask_cvtne2ph_phf8(__m128i __W, __mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtne2ph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2hf8128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  // CHECK: ret <2 x i64> %{{.*}}
  return _mm_mask_cvtne2ph_phf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtne2ph_phf8(__mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtne2ph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2hf8128(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_cvtne2ph_phf8(__U, __A, __B);
}

__m256i test_mm256_cvtne2ph_phf8(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtne2ph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2hf8256(
  return _mm256_cvtne2ph_phf8(__A, __B);
}

__m256i test_mm256_mask_cvtne2ph_phf8(__m256i __W, __mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtne2ph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2hf8256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  // CHECK: ret <4 x i64> %{{.*}}
  return _mm256_mask_cvtne2ph_phf8(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_cvtne2ph_phf8(__mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtne2ph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2hf8256(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_cvtne2ph_phf8(__U, __A, __B);
}

__m128i test_mm_cvtnes2ph_phf8(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cvtnes2ph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s128(
  return _mm_cvtnes2ph_phf8(__A, __B);
}

__m128i test_mm_mask_cvtnes2ph_phf8(__m128i __W, __mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cvtnes2ph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  // CHECK: ret <2 x i64> %{{.*}}
  return _mm_mask_cvtnes2ph_phf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtnes2ph_phf8(__mmask16 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtnes2ph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s128(
  // CHECK: zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_cvtnes2ph_phf8(__U, __A, __B);
}

__m256i test_mm256_cvtnes2ph_phf8(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cvtnes2ph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s256(
  return _mm256_cvtnes2ph_phf8(__A, __B);
}

__m256i test_mm256_mask_cvtnes2ph_phf8(__m256i __W, __mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtnes2ph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  // CHECK: ret <4 x i64> %{{.*}}
  return _mm256_mask_cvtnes2ph_phf8(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_cvtnes2ph_phf8(__mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtnes2ph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s256(
  // CHECK: zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_cvtnes2ph_phf8(__U, __A, __B);
}

__m128h test_mm_cvtnehf8_ph(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtnehf8_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvthf82ph128(
  return _mm_cvtnehf8_ph(__A);
}

__m128h test_mm_mask_cvtnehf8_ph(__m128h __A, __mmask8 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_cvtnehf8_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvthf82ph128(
  return _mm_mask_cvtnehf8_ph(__A, __B, __C);
}

__m128h test_mm_maskz_cvtnehf8_ph(__mmask8 __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtnehf8_ph(
  // CHECK: call <8 x half> @llvm.x86.avx10.mask.vcvthf82ph128(
  return _mm_maskz_cvtnehf8_ph(__A, __B);
}

__m256h test_mm256_cvtnehf8_ph(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvtnehf8_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvthf82ph256(
  return _mm256_cvtnehf8_ph(__A);
}

__m256h test_mm256_mask_cvtnehf8_ph(__m256h __A, __mmask16 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtnehf8_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvthf82ph256(
  return _mm256_mask_cvtnehf8_ph(__A, __B, __C);
}

__m256h test_mm256_maskz_cvtnehf8_ph(__mmask16 __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtnehf8_ph(
  // CHECK: call <16 x half> @llvm.x86.avx10.mask.vcvthf82ph256(
  return _mm256_maskz_cvtnehf8_ph(__A, __B);
}

__m128i test_mm_cvtneph_pbf8(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtneph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8128(
  return _mm_cvtneph_pbf8(__A);
}

__m128i test_mm_mask_cvtneph_pbf8(__m128i __A, __mmask8 __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_cvtneph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8128(
  return _mm_mask_cvtneph_pbf8(__A, __B, __C);
}

__m128i test_mm_maskz_cvtneph_pbf8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtneph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8128(
  return _mm_maskz_cvtneph_pbf8(__A, __B);
}

__m128i test_mm256_cvtneph_pbf8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtneph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8256(
  return _mm256_cvtneph_pbf8(__A);
}

__m128i test_mm256_mask_cvtneph_pbf8(__m128i __A, __mmask16 __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtneph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8256(
  return _mm256_mask_cvtneph_pbf8(__A, __B, __C);
}

__m128i test_mm256_maskz_cvtneph_pbf8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtneph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8256(
  return _mm256_maskz_cvtneph_pbf8(__A, __B);
}

__m128i test_mm_cvtnesph_pbf8(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtnesph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s128(
  return _mm_cvtnesph_pbf8(__A);
}

__m128i test_mm_mask_cvtnesph_pbf8(__m128i __A, __mmask8 __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_cvtnesph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s128(
  return _mm_mask_cvtnesph_pbf8(__A, __B, __C);
}

__m128i test_mm_maskz_cvtnesph_pbf8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtnesph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s128(
  return _mm_maskz_cvtnesph_pbf8(__A, __B);
}

__m128i test_mm256_cvtnesph_pbf8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtnesph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s256(
  return _mm256_cvtnesph_pbf8(__A);
}

__m128i test_mm256_mask_cvtnesph_pbf8(__m128i __A, __mmask16 __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtnesph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s256(
  return _mm256_mask_cvtnesph_pbf8(__A, __B, __C);
}

__m128i test_mm256_maskz_cvtnesph_pbf8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtnesph_pbf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s256(
  return _mm256_maskz_cvtnesph_pbf8(__A, __B);
}

__m128i test_mm_cvtneph_phf8(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtneph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8128(
  return _mm_cvtneph_phf8(__A);
}

__m128i test_mm_mask_cvtneph_phf8(__m128i __A, __mmask8 __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_cvtneph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8128(
  return _mm_mask_cvtneph_phf8(__A, __B, __C);
}

__m128i test_mm_maskz_cvtneph_phf8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtneph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8128(
  return _mm_maskz_cvtneph_phf8(__A, __B);
}

__m128i test_mm256_cvtneph_phf8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtneph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8256(
  return _mm256_cvtneph_phf8(__A);
}

__m128i test_mm256_mask_cvtneph_phf8(__m128i __A, __mmask16 __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtneph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8256(
  return _mm256_mask_cvtneph_phf8(__A, __B, __C);
}

__m128i test_mm256_maskz_cvtneph_phf8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtneph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8256(
  return _mm256_maskz_cvtneph_phf8(__A, __B);
}

__m128i test_mm_cvtnesph_phf8(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtnesph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s128(
  return _mm_cvtnesph_phf8(__A);
}

__m128i test_mm_mask_cvtnesph_phf8(__m128i __A, __mmask8 __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_cvtnesph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s128(
  return _mm_mask_cvtnesph_phf8(__A, __B, __C);
}

__m128i test_mm_maskz_cvtnesph_phf8(__mmask8 __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtnesph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s128(
  return _mm_maskz_cvtnesph_phf8(__A, __B);
}

__m128i test_mm256_cvtnesph_phf8(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtnesph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s256(
  return _mm256_cvtnesph_phf8(__A);
}

__m128i test_mm256_mask_cvtnesph_phf8(__m128i __A, __mmask16 __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_cvtnesph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s256(
  return _mm256_mask_cvtnesph_phf8(__A, __B, __C);
}

__m128i test_mm256_maskz_cvtnesph_phf8(__mmask16 __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtnesph_phf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s256(
  return _mm256_maskz_cvtnesph_phf8(__A, __B);
}

__m256h test_mm256_cvtpbf8_ph(__m128i A) {
  // CHECK-LABEL: @test_mm256_cvtpbf8_ph
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: ret <16 x half> %{{.*}}
  return _mm256_cvtpbf8_ph(A);
}

__m256h test_mm256_mask_cvtpbf8_ph(__m256h S, __mmask16 M, __m128i A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpbf8_ph
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  // CHECK: ret <16 x half> %{{.*}}
  return _mm256_mask_cvtpbf8_ph(S, M, A);
}

__m256h test_mm256_maskz_cvtpbf8_ph(__mmask16 M, __m128i A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpbf8_ph
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: ret <16 x half> %{{.*}}
  return _mm256_maskz_cvtpbf8_ph(M, A);
}

__m128h test_mm_cvtpbf8_ph(__m128i A) {
  // CHECK-LABEL: @test_mm_cvtpbf8_ph
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: ret <8 x half> %{{.*}}
  return _mm_cvtpbf8_ph(A);
}

__m128h test_mm_mask_cvtpbf8_ph(__m128h S, __mmask8 M, __m128i A) {
  // CHECK-LABEL: @test_mm_mask_cvtpbf8_ph
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  // CHECK: ret <8 x half> %{{.*}}
  return _mm_mask_cvtpbf8_ph(S, M, A);
}

__m128h test_mm_maskz_cvtpbf8_ph(__mmask8 M, __m128i A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpbf8_ph
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: ret <8 x half> %{{.*}}
  return _mm_maskz_cvtpbf8_ph(M, A);
}
