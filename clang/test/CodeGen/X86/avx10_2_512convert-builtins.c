// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2-512 \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2-512 \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512h test_mm512_cvtx2ps_ph(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvtx2ps_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.512
  return _mm512_cvtx2ps_ph(__A, __B);
}

__m512h test_mm512_mask_cvtx2ps_ph(__m512h __W, __mmask32 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtx2ps_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.512
  return _mm512_mask_cvtx2ps_ph(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_cvtx2ps_ph(__mmask32 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtx2ps_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.512
  return _mm512_maskz_cvtx2ps_ph(__U, __A, __B);
}

__m512h test_mm512_cvtx_round2ps_ph(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvtx_round2ps_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.512
  return _mm512_cvtx_round2ps_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_cvtx_round2ps_ph(__m512h __W, __mmask32 __U, __m512 __A, __m512 __B) {
// CHECK-LABEL: @test_mm512_mask_cvtx_round2ps_ph(
// CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.512
  return _mm512_mask_cvtx_round2ps_ph(__W, __U, __A, __B, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_cvtx_round2ps_ph(__mmask32 __U, __m512 __A, __m512 __B) {
// CHECK-LABEL: @test_mm512_maskz_cvtx_round2ps_ph(
// CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvt2ps2phx.512
  return _mm512_maskz_cvtx_round2ps_ph(__U, __A, __B, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

__m256i test_mm512_cvtbiasph_pbf8(__m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_cvtbiasph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8512(
  return _mm512_cvtbiasph_pbf8(__A, __B);
}

__m256i test_mm512_mask_cvtbiasph_pbf8(__m256i __W, __mmask32 __U, __m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtbiasph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8512(
  return _mm512_mask_cvtbiasph_pbf8(__W, __U, __A, __B);
}

__m256i test_mm512_maskz_cvtbiasph_pbf8(__mmask32 __U, __m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbiasph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8512(
  return _mm512_maskz_cvtbiasph_pbf8(__U, __A, __B);
}

__m256i test_mm512_cvtbiassph_pbf8(__m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_cvtbiassph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s512(
  return _mm512_cvtbiassph_pbf8(__A, __B);
}

__m256i test_mm512_mask_cvtbiassph_pbf8(__m256i __W, __mmask32 __U, __m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtbiassph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s512(
  return _mm512_mask_cvtbiassph_pbf8(__W, __U, __A, __B);
}

__m256i test_mm512_maskz_cvtbiassph_pbf8(__mmask32 __U, __m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbiassph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2bf8s512(
  return _mm512_maskz_cvtbiassph_pbf8(__U, __A, __B);
}

__m256i test_mm512_cvtbiasph_phf8(__m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_cvtbiasph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8512(
  return _mm512_cvtbiasph_phf8(__A, __B);
}

__m256i test_mm512_mask_cvtbiasph_phf8(__m256i __W, __mmask32 __U, __m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtbiasph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8512(
  return _mm512_mask_cvtbiasph_phf8(__W, __U, __A, __B);
}

__m256i test_mm512_maskz_cvtbiasph_phf8(__mmask32 __U, __m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbiasph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8512(
  return _mm512_maskz_cvtbiasph_phf8(__U, __A, __B);
}

__m256i test_mm512_cvtbiassph_phf8(__m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_cvtbiassph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s512(
  return _mm512_cvtbiassph_phf8(__A, __B);
}

__m256i test_mm512_mask_cvtbiassph_phf8(__m256i __W, __mmask32 __U, __m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtbiassph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s512(
  return _mm512_mask_cvtbiassph_phf8(__W, __U, __A, __B);
}

__m256i test_mm512_maskz_cvtbiassph_phf8(__mmask32 __U, __m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbiassph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtbiasph2hf8s512(
  return _mm512_maskz_cvtbiassph_phf8(__U, __A, __B);
}

__m512i test_mm512_cvtne2ph_pbf8(__m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_cvtne2ph_pbf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2bf8512(
  return _mm512_cvtne2ph_pbf8(__A, __B);
}

__m512i test_mm512_mask_cvtne2ph_pbf8(__m512i __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtne2ph_pbf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2bf8512(
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  // CHECK: ret <8 x i64> %{{.*}}
  return _mm512_mask_cvtne2ph_pbf8(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_cvtne2ph_pbf8(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtne2ph_pbf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2bf8512(
  // CHECK: zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_cvtne2ph_pbf8(__U, __A, __B);
}

__m512i test_mm512_cvtnes2ph_pbf8(__m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_cvtnes2ph_pbf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s512(
  return _mm512_cvtnes2ph_pbf8(__A, __B);
}

__m512i test_mm512_mask_cvtnes2ph_pbf8(__m512i __W, __mmask64 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtnes2ph_pbf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s512(
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  // CHECK: ret <8 x i64> %{{.*}}
  return _mm512_mask_cvtnes2ph_pbf8(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_cvtnes2ph_pbf8(__mmask64 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtnes2ph_pbf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2bf8s512(
  // CHECK: zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_cvtnes2ph_pbf8(__U, __A, __B);
}

__m512i test_mm512_cvtne2ph_phf8(__m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_cvtne2ph_phf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2hf8512(
  return _mm512_cvtne2ph_phf8(__A, __B);
}

__m512i test_mm512_mask_cvtne2ph_phf8(__m512i __W, __mmask64 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtne2ph_phf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2hf8512(
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  // CHECK: ret <8 x i64> %{{.*}}
  return _mm512_mask_cvtne2ph_phf8(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_cvtne2ph_phf8(__mmask64 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtne2ph_phf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2hf8512(
  // CHECK: zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_cvtne2ph_phf8(__U, __A, __B);
}

__m512i test_mm512_cvtnes2ph_phf8(__m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_cvtnes2ph_phf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s512(
  return _mm512_cvtnes2ph_phf8(__A, __B);
}

__m512i test_mm512_mask_cvtnes2ph_phf8(__m512i __W, __mmask64 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtnes2ph_phf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s512(
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  // CHECK: ret <8 x i64> %{{.*}}
  return _mm512_mask_cvtnes2ph_phf8(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_cvtnes2ph_phf8(__mmask64 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtnes2ph_phf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtne2ph2hf8s512(
  // CHECK: zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_cvtnes2ph_phf8(__U, __A, __B);
}

__m512h test_mm512_cvtnehf8_ph(__m256i __A) {
  // CHECK-LABEL: @test_mm512_cvtnehf8_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvthf82ph512(
  return _mm512_cvtnehf8_ph(__A);
}

__m512h test_mm512_mask_cvtnehf8_ph(__m512h __A, __mmask32 __B, __m256i __C) {
  // CHECK-LABEL: @test_mm512_mask_cvtnehf8_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvthf82ph512(
  return _mm512_mask_cvtnehf8_ph(__A, __B, __C);
}

__m512h test_mm512_maskz_cvtnehf8_ph(__mmask32 __A, __m256i __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtnehf8_ph(
  // CHECK: call <32 x half> @llvm.x86.avx10.mask.vcvthf82ph512(
  return _mm512_maskz_cvtnehf8_ph(__A, __B);
}

__m256i test_mm512_cvtneph_pbf8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_cvtneph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8512(
  return _mm512_cvtneph_pbf8(__A);
}

__m256i test_mm512_mask_cvtneph_pbf8(__m256i __A, __mmask32 __B, __m512h __C) {
  // CHECK-LABEL: @test_mm512_mask_cvtneph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8512(
  return _mm512_mask_cvtneph_pbf8(__A, __B, __C);
}

__m256i test_mm512_maskz_cvtneph_pbf8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtneph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8512(
  return _mm512_maskz_cvtneph_pbf8(__A, __B);
}

__m256i test_mm512_cvtnesph_pbf8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_cvtnesph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s512(
  return _mm512_cvtnesph_pbf8(__A);
}

__m256i test_mm512_mask_cvtnesph_pbf8(__m256i __A, __mmask32 __B, __m512h __C) {
  // CHECK-LABEL: @test_mm512_mask_cvtnesph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s512(
  return _mm512_mask_cvtnesph_pbf8(__A, __B, __C);
}

__m256i test_mm512_maskz_cvtnesph_pbf8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtnesph_pbf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2bf8s512(
  return _mm512_maskz_cvtnesph_pbf8(__A, __B);
}

__m256i test_mm512_cvtneph_phf8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_cvtneph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8512(
  return _mm512_cvtneph_phf8(__A);
}

__m256i test_mm512_mask_cvtneph_phf8(__m256i __A, __mmask32 __B, __m512h __C) {
  // CHECK-LABEL: @test_mm512_mask_cvtneph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8512(
  return _mm512_mask_cvtneph_phf8(__A, __B, __C);
}

__m256i test_mm512_maskz_cvtneph_phf8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtneph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8512(
  return _mm512_maskz_cvtneph_phf8(__A, __B);
}

__m256i test_mm512_cvtnesph_phf8(__m512h __A) {
  // CHECK-LABEL: @test_mm512_cvtnesph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s512(
  return _mm512_cvtnesph_phf8(__A);
}

__m256i test_mm512_mask_cvtnesph_phf8(__m256i __A, __mmask32 __B, __m512h __C) {
  // CHECK-LABEL: @test_mm512_mask_cvtnesph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s512(
  return _mm512_mask_cvtnesph_phf8(__A, __B, __C);
}

__m256i test_mm512_maskz_cvtnesph_phf8(__mmask32 __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtnesph_phf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.mask.vcvtneph2hf8s512(
  return _mm512_maskz_cvtnesph_phf8(__A, __B);
}

__m512h test_mm512_cvtpbf8_ph(__m256i A) {
  // CHECK-LABEL: @test_mm512_cvtpbf8_ph
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: @llvm.x86.avx512.pslli.w.512
  // CHECK: ret <32 x half> %{{.*}}
  return _mm512_cvtpbf8_ph(A);
}

__m512h test_mm512_mask_cvtpbf8_ph(__m512h S, __mmask16 M, __m256i A) {
  // CHECK-LABEL: @test_mm512_mask_cvtpbf8_ph
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: @llvm.x86.avx512.pslli.w.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  // CHECK: ret <32 x half> %{{.*}}
  return _mm512_mask_cvtpbf8_ph(S, M, A);
}

__m512h test_mm512_maskz_cvtpbf8_ph(__mmask16 M, __m256i A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtpbf8_ph
  // CHECK: sext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  // CHECK: @llvm.x86.avx512.pslli.w.512
  // CHECK: ret <32 x half> %{{.*}}
  return _mm512_maskz_cvtpbf8_ph(M, A);
}
