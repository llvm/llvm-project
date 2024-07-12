// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

__m128i test_mm_broadcastmb_epi64(__m128i a,__m128i b) {
  // CHECK-LABEL: @test_mm_broadcastmb_epi64
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: zext i8 %{{.*}} to i64
  // CHECK: insertelement <2 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  return _mm_broadcastmb_epi64(_mm_cmpeq_epi32_mask (a, b)); 
}

__m256i test_mm256_broadcastmb_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: @test_mm256_broadcastmb_epi64
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: zext i8 %{{.*}} to i64
  // CHECK: insertelement <4 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  return _mm256_broadcastmb_epi64(_mm256_cmpeq_epi64_mask ( a, b)); 
}

__m128i test_mm_broadcastmw_epi32(__m512i a, __m512i b) {
  // CHECK-LABEL: @test_mm_broadcastmw_epi32
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: zext i16 %{{.*}} to i32
  // CHECK: insertelement <4 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  return _mm_broadcastmw_epi32(_mm512_cmpeq_epi32_mask ( a, b));
}

__m256i test_mm256_broadcastmw_epi32(__m512i a, __m512i b) {
  // CHECK-LABEL: @test_mm256_broadcastmw_epi32
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: zext i16 %{{.*}} to i32
  // CHECK: insertelement <8 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  return _mm256_broadcastmw_epi32(_mm512_cmpeq_epi32_mask ( a, b)); 
}

__m128i test_mm_conflict_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_conflict_epi64
  // CHECK: @llvm.x86.avx512.conflict.q.128
  return _mm_conflict_epi64(__A); 
}

__m128i test_mm_mask_conflict_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_conflict_epi64
  // CHECK: @llvm.x86.avx512.conflict.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_conflict_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_conflict_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_conflict_epi64
  // CHECK: @llvm.x86.avx512.conflict.q.128
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_conflict_epi64(__U, __A); 
}

__m256i test_mm256_conflict_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_conflict_epi64
  // CHECK: @llvm.x86.avx512.conflict.q.256
  return _mm256_conflict_epi64(__A); 
}

__m256i test_mm256_mask_conflict_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_conflict_epi64
  // CHECK: @llvm.x86.avx512.conflict.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_conflict_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_conflict_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_conflict_epi64
  // CHECK: @llvm.x86.avx512.conflict.q.256
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_conflict_epi64(__U, __A); 
}

__m128i test_mm_conflict_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_conflict_epi32
  // CHECK: @llvm.x86.avx512.conflict.d.128
  return _mm_conflict_epi32(__A); 
}

__m128i test_mm_mask_conflict_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_conflict_epi32
  // CHECK: @llvm.x86.avx512.conflict.d.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_conflict_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_conflict_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_conflict_epi32
  // CHECK: @llvm.x86.avx512.conflict.d.128
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_conflict_epi32(__U, __A); 
}

__m256i test_mm256_conflict_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_conflict_epi32
  // CHECK: @llvm.x86.avx512.conflict.d.256
  return _mm256_conflict_epi32(__A); 
}

__m256i test_mm256_mask_conflict_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_conflict_epi32
  // CHECK: @llvm.x86.avx512.conflict.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_conflict_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_conflict_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_conflict_epi32
  // CHECK: @llvm.x86.avx512.conflict.d.256
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
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
