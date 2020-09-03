//  RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin \
//  RUN:            -target-feature +avx512bf16 -emit-llvm -o - -Wall -Werror \
//  RUN:            | FileCheck %s

#include <immintrin.h>

float test_mm_cvtsbh_ss(__bfloat16 A) {
  // CHECK-LABEL: @test_mm_cvtsbh_ss
  // CHECK: zext i16 %{{.*}} to i32
  // CHECK: shl i32 %{{.*}}, 16
  // CHECK: bitcast i32 %{{.*}} to float
  // CHECK: ret float %{{.*}}
  return _mm_cvtsbh_ss(A);
}

__m512bh test_mm512_cvtne2ps_pbh(__m512 A, __m512 B) {
  // CHECK-LABEL: @test_mm512_cvtne2ps_pbh
  // CHECK: @llvm.x86.avx512bf16.cvtne2ps2bf16.512
  // CHECK: ret <32 x i16> %{{.*}}
  return _mm512_cvtne2ps_pbh(A, B);
}

__m512bh test_mm512_maskz_cvtne2ps_pbh(__m512 A, __m512 B, __mmask32 U) {
  // CHECK-LABEL: @test_mm512_maskz_cvtne2ps_pbh
  // CHECK: @llvm.x86.avx512bf16.cvtne2ps2bf16.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  // CHECK: ret <32 x i16> %{{.*}}
  return _mm512_maskz_cvtne2ps_pbh(U, A, B);
}

__m512bh test_mm512_mask_cvtne2ps_pbh(__m512bh C, __mmask32 U, __m512 A, __m512 B) {
  // CHECK-LABEL: @test_mm512_mask_cvtne2ps_pbh
  // CHECK: @llvm.x86.avx512bf16.cvtne2ps2bf16.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  // CHECK: ret <32 x i16> %{{.*}}
  return _mm512_mask_cvtne2ps_pbh(C, U, A, B);
}

__m256bh test_mm512_cvtneps_pbh(__m512 A) {
  // CHECK-LABEL: @test_mm512_cvtneps_pbh
  // CHECK: @llvm.x86.avx512bf16.cvtneps2bf16.512
  // CHECK: ret <16 x i16> %{{.*}}
  return _mm512_cvtneps_pbh(A);
}

__m256bh test_mm512_mask_cvtneps_pbh(__m256bh C, __mmask16 U, __m512 A) {
  // CHECK-LABEL: @test_mm512_mask_cvtneps_pbh
  // CHECK: @llvm.x86.avx512bf16.cvtneps2bf16.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  // CHECK: ret <16 x i16> %{{.*}}
  return _mm512_mask_cvtneps_pbh(C, U, A);
}

__m256bh test_mm512_maskz_cvtneps_pbh(__m512 A, __mmask16 U) {
  // CHECK-LABEL: @test_mm512_maskz_cvtneps_pbh
  // CHECK: @llvm.x86.avx512bf16.cvtneps2bf16.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  // CHECK: ret <16 x i16> %{{.*}}
  return _mm512_maskz_cvtneps_pbh(U, A);
}

__m512 test_mm512_dpbf16_ps(__m512 D, __m512bh A, __m512bh B) {
  // CHECK-LABEL: @test_mm512_dpbf16_ps
  // CHECK: @llvm.x86.avx512bf16.dpbf16ps.512
  // CHECK: ret <16 x float> %{{.*}}
  return _mm512_dpbf16_ps(D, A, B);
}

__m512 test_mm512_maskz_dpbf16_ps(__m512 D, __m512bh A, __m512bh B, __mmask16 U) {
  // CHECK-LABEL: @test_mm512_maskz_dpbf16_ps
  // CHECK: @llvm.x86.avx512bf16.dpbf16ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  // CHECK: ret <16 x float> %{{.*}}
  return _mm512_maskz_dpbf16_ps(U, D, A, B);
}

__m512 test_mm512_mask_dpbf16_ps(__m512 D, __m512bh A, __m512bh B, __mmask16 U) {
  // CHECK-LABEL: @test_mm512_mask_dpbf16_ps
  // CHECK: @llvm.x86.avx512bf16.dpbf16ps.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  // CHECK: ret <16 x float> %{{.*}}
  return _mm512_mask_dpbf16_ps(D, U, A, B);
}

__m512 test_mm512_cvtpbh_ps(__m256bh A) {
  // CHECK-LABEL: @test_mm512_cvtpbh_ps
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: @llvm.x86.avx512.pslli.d.512
  // CHECK: bitcast <8 x i64> %{{.*}} to <16 x float>
  // CHECK: ret <16 x float> %{{.*}}
  return _mm512_cvtpbh_ps(A);
}

__m512 test_mm512_maskz_cvtpbh_ps(__mmask16 M, __m256bh A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtpbh_ps
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  // CHECK: @llvm.x86.avx512.pslli.d.512
  // CHECK: bitcast <8 x i64> %{{.*}} to <16 x float>
  // CHECK: ret <16 x float> %{{.*}}
  return _mm512_maskz_cvtpbh_ps(M, A);
}

__m512 test_mm512_mask_cvtpbh_ps(__m512 S, __mmask16 M, __m256bh A) {
  // CHECK-LABEL: @test_mm512_mask_cvtpbh_ps
  // CHECK: sext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: @llvm.x86.avx512.pslli.d.512
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  // CHECK: bitcast <8 x i64> %{{.*}} to <16 x float>
  // CHECK: ret <16 x float> %{{.*}}
  return _mm512_mask_cvtpbh_ps(S, M, A);
}
