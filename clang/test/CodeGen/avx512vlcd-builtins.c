// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

__m128i test_mm_broadcastmb_epi64(__mmask8 __A) {
  // CHECK-LABEL: @test_mm_broadcastmb_epi64
  // CHECK: @llvm.x86.avx512.broadcastmb.128
  return _mm_broadcastmb_epi64(__A); 
}

__m256i test_mm256_broadcastmb_epi64(__mmask8 __A) {
  // CHECK-LABEL: @test_mm256_broadcastmb_epi64
  // CHECK: @llvm.x86.avx512.broadcastmb.256
  return _mm256_broadcastmb_epi64(__A); 
}

__m128i test_mm_broadcastmw_epi32(__mmask16 __A) {
  // CHECK-LABEL: @test_mm_broadcastmw_epi32
  // CHECK: @llvm.x86.avx512.broadcastmw.128
  return _mm_broadcastmw_epi32(__A); 
}

__m256i test_mm256_broadcastmw_epi32(__mmask16 __A) {
  // CHECK-LABEL: @test_mm256_broadcastmw_epi32
  // CHECK: @llvm.x86.avx512.broadcastmw.256
  return _mm256_broadcastmw_epi32(__A); 
}

__m128i test_mm_conflict_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.128
  return _mm_conflict_epi64(__A); 
}

__m128i test_mm_mask_conflict_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.128
  return _mm_mask_conflict_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_conflict_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.128
  return _mm_maskz_conflict_epi64(__U, __A); 
}

__m256i test_mm256_conflict_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.256
  return _mm256_conflict_epi64(__A); 
}

__m256i test_mm256_mask_conflict_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.256
  return _mm256_mask_conflict_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_conflict_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.256
  return _mm256_maskz_conflict_epi64(__U, __A); 
}

__m128i test_mm_conflict_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.128
  return _mm_conflict_epi32(__A); 
}

__m128i test_mm_mask_conflict_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.128
  return _mm_mask_conflict_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_conflict_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.128
  return _mm_maskz_conflict_epi32(__U, __A); 
}

__m256i test_mm256_conflict_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.256
  return _mm256_conflict_epi32(__A); 
}

__m256i test_mm256_mask_conflict_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.256
  return _mm256_mask_conflict_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_conflict_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.256
  return _mm256_maskz_conflict_epi32(__U, __A); 
}

__m128i test_mm_lzcnt_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_lzcnt_epi32
  // CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 false)
  return _mm_lzcnt_epi32(__A); 
}

__m128i test_mm_mask_lzcnt_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_lzcnt_epi32
  // CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 false)
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_lzcnt_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_lzcnt_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_lzcnt_epi32
  // CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 false)
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_lzcnt_epi32(__U, __A); 
}

__m256i test_mm256_lzcnt_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_lzcnt_epi32
  // CHECK: call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %{{.*}}, i1 false)
  return _mm256_lzcnt_epi32(__A); 
}

__m256i test_mm256_mask_lzcnt_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_lzcnt_epi32
  // CHECK: call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %{{.*}}, i1 false)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_lzcnt_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_lzcnt_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_lzcnt_epi32
  // CHECK: call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %{{.*}}, i1 false)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_lzcnt_epi32(__U, __A); 
}

__m128i test_mm_lzcnt_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_lzcnt_epi64
  // CHECK: call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 false)
  return _mm_lzcnt_epi64(__A); 
}

__m128i test_mm_mask_lzcnt_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_lzcnt_epi64
  // CHECK: call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 false)
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_lzcnt_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_lzcnt_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_lzcnt_epi64
  // CHECK: call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 false)
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_lzcnt_epi64(__U, __A); 
}

__m256i test_mm256_lzcnt_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_lzcnt_epi64
  // CHECK: call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %{{.*}}, i1 false)
  return _mm256_lzcnt_epi64(__A); 
}

__m256i test_mm256_mask_lzcnt_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_lzcnt_epi64
  // CHECK: call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %{{.*}}, i1 false)
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_lzcnt_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_lzcnt_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_lzcnt_epi64
  // CHECK: call <4 x i64> @llvm.ctlz.v4i64(<4 x i64> %{{.*}}, i1 false)
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_lzcnt_epi64(__U, __A); 
}
