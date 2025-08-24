// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bf16 -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bf16 -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bf16 -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512bf16 -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128bh test_mm_cvtne2ps2bf16(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_cvtne2ps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_cvtne2ps_pbh(A, B);
}

__m128bh test_mm_maskz_cvtne2ps2bf16(__m128 A, __m128 B, __mmask8 U) {
  // CHECK-LABEL: test_mm_maskz_cvtne2ps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_cvtne2ps_pbh(U, A, B);
}

__m128bh test_mm_mask_cvtne2ps2bf16(__m128bh C, __mmask8 U, __m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_mask_cvtne2ps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask_cvtne2ps_pbh(C, U, A, B);
}

__m256bh test_mm256_cvtne2ps2bf16(__m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_cvtne2ps2bf16
  // CHECK: call {{.*}}<16 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_cvtne2ps_pbh(A, B);
}

__m256bh test_mm256_maskz_cvtne2ps2bf16(__m256 A, __m256 B, __mmask16 U) {
  // CHECK-LABEL: test_mm256_maskz_cvtne2ps2bf16
  // CHECK: call {{.*}}<16 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_cvtne2ps_pbh(U, A, B);
}

__m256bh test_mm256_mask_cvtne2ps2bf16(__m256bh C, __mmask16 U, __m256 A, __m256 B) {
  // CHECK-LABEL: test_mm256_mask_cvtne2ps2bf16
  // CHECK: call {{.*}}<16 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask_cvtne2ps_pbh(C, U, A, B);
}

__m512bh test_mm512_cvtne2ps2bf16(__m512 A, __m512 B) {
  // CHECK-LABEL: test_mm512_cvtne2ps2bf16
  // CHECK: call {{.*}}<32 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_cvtne2ps_pbh(A, B);
}

__m512bh test_mm512_maskz_cvtne2ps2bf16(__m512 A, __m512 B, __mmask32 U) {
  // CHECK-LABEL: test_mm512_maskz_cvtne2ps2bf16
  // CHECK: call {{.*}}<32 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_cvtne2ps_pbh(U, A, B);
}

__m512bh test_mm512_mask_cvtne2ps2bf16(__m512bh C, __mmask32 U, __m512 A, __m512 B) {
  // CHECK-LABEL: test_mm512_mask_cvtne2ps2bf16
  // CHECK: call {{.*}}<32 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %{{.*}}, <16 x float> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_cvtne2ps_pbh(C, U, A, B);
}

__m128bh test_mm_cvtneps2bf16(__m128 A) {
  // CHECK-LABEL: test_mm_cvtneps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.vcvtneps2bf16128(<4 x float> %{{.*}})
  return _mm_cvtneps_pbh(A);
}

__m128bh test_mm_mask_cvtneps2bf16(__m128bh C, __mmask8 U, __m128 A) {
  // CHECK-LABEL: test_mm_mask_cvtneps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %{{.*}}, <8 x bfloat> %{{.*}}, <4 x i1> %{{.*}})
  return _mm_mask_cvtneps_pbh(C, U, A);
}

__m128bh test_mm_maskz_cvtneps2bf16(__m128 A, __mmask8 U) {
  // CHECK-LABEL: test_mm_maskz_cvtneps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %{{.*}}, <8 x bfloat> %{{.*}}, <4 x i1> %{{.*}})
  return _mm_maskz_cvtneps_pbh(U, A);
}

__m128bh test_mm256_cvtneps2bf16(__m256 A) {
  // CHECK-LABEL: test_mm256_cvtneps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.vcvtneps2bf16256(<8 x float> %{{.*}})
  return _mm256_cvtneps_pbh(A);
}

__m128bh test_mm256_mask_cvtneps2bf16(__m128bh C, __mmask8 U, __m256 A) {
  // CHECK-LABEL: test_mm256_mask_cvtneps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm256_mask_cvtneps_pbh(C, U, A);
}

__m128bh test_mm256_maskz_cvtneps2bf16(__m256 A, __mmask8 U) {
  // CHECK-LABEL: test_mm256_maskz_cvtneps2bf16
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm256_maskz_cvtneps_pbh(U, A);
}

__m128 test_mm_dpbf16_ps(__m128 D, __m128bh A, __m128bh B) {
  // CHECK-LABEL: test_mm_dpbf16_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_dpbf16_ps(D, A, B);
}

__m128 test_mm_maskz_dpbf16_ps(__m128 D, __m128bh A, __m128bh B, __mmask8 U) {
  // CHECK-LABEL: test_mm_maskz_dpbf16_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_dpbf16_ps(U, D, A, B);
}

__m128 test_mm_mask_dpbf16_ps(__m128 D, __m128bh A, __m128bh B, __mmask8 U) {
  // CHECK-LABEL: test_mm_mask_dpbf16_ps
  // CHECK: call {{.*}}<4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_dpbf16_ps(D, U, A, B);
}

__m256 test_mm256_dpbf16_ps(__m256 D, __m256bh A, __m256bh B) {
  // CHECK-LABEL: test_mm256_dpbf16_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  return _mm256_dpbf16_ps(D, A, B);
}

__m256 test_mm256_maskz_dpbf16_ps(__m256 D, __m256bh A, __m256bh B, __mmask8 U) {
  // CHECK-LABEL: test_mm256_maskz_dpbf16_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_dpbf16_ps(U, D, A, B);
}

__m256 test_mm256_mask_dpbf16_ps(__m256 D, __m256bh A, __m256bh B, __mmask8 U) {
  // CHECK-LABEL: test_mm256_mask_dpbf16_ps
  // CHECK: call {{.*}}<8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_dpbf16_ps(D, U, A, B);
}

__bf16 test_mm_cvtness_sbh(float A) {
  // CHECK-LABEL: test_mm_cvtness_sbh
  // CHECK: call {{.*}}<8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %{{.*}}, <8 x bfloat> %{{.*}}, <4 x i1> splat (i1 true))
  return _mm_cvtness_sbh(A);
}

__m128 test_mm_cvtpbh_ps(__m128bh A) {
  // CHECK-LABEL: test_mm_cvtpbh_ps
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i32>
  // CHECK: call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_cvtpbh_ps(A);
}

__m256 test_mm256_cvtpbh_ps(__m128bh A) {
  // CHECK-LABEL: test_mm256_cvtpbh_ps
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: call <8 x i32> @llvm.x86.avx2.pslli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_cvtpbh_ps(A);
}

__m128 test_mm_maskz_cvtpbh_ps(__mmask8 M, __m128bh A) {
  // CHECK-LABEL: test_mm_maskz_cvtpbh_ps
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i32>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  // CHECK: call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  return _mm_maskz_cvtpbh_ps(M, A);
}

__m256 test_mm256_maskz_cvtpbh_ps(__mmask8 M, __m128bh A) {
  // CHECK-LABEL: test_mm256_maskz_cvtpbh_ps
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  // CHECK: call <8 x i32> @llvm.x86.avx2.pslli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  return _mm256_maskz_cvtpbh_ps(M, A);
}

__m128 test_mm_mask_cvtpbh_ps(__m128 S, __mmask8 M, __m128bh A) {
  // CHECK-LABEL: test_mm_mask_cvtpbh_ps
  // CHECK: sext <4 x i16> %{{.*}} to <4 x i32>
  // CHECK: call <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_cvtpbh_ps(S, M, A);
}

__m256 test_mm256_mask_cvtpbh_ps(__m256 S, __mmask8 M, __m128bh A) {
  // CHECK-LABEL: test_mm256_mask_cvtpbh_ps
  // CHECK: sext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: call <8 x i32> @llvm.x86.avx2.pslli.d(<8 x i32> %{{.*}}, i32 %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_cvtpbh_ps(S, M, A);
}
