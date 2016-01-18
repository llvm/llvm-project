// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512cd -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m512i test_mm512_conflict_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.512
  return _mm512_conflict_epi64(__A); 
}
__m512i test_mm512_mask_conflict_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.512
  return _mm512_mask_conflict_epi64(__W,__U,__A); 
}
__m512i test_mm512_maskz_conflict_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_conflict_epi64
  // CHECK: @llvm.x86.avx512.mask.conflict.q.512
  return _mm512_maskz_conflict_epi64(__U,__A); 
}
__m512i test_mm512_conflict_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.512
  return _mm512_conflict_epi32(__A); 
}
__m512i test_mm512_mask_conflict_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.512
  return _mm512_mask_conflict_epi32(__W,__U,__A); 
}
__m512i test_mm512_maskz_conflict_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_conflict_epi32
  // CHECK: @llvm.x86.avx512.mask.conflict.d.512
  return _mm512_maskz_conflict_epi32(__U,__A); 
}
__m512i test_mm512_lzcnt_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_lzcnt_epi32
  // CHECK: @llvm.x86.avx512.mask.lzcnt.d.512
  return _mm512_lzcnt_epi32(__A); 
}
__m512i test_mm512_mask_lzcnt_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_lzcnt_epi32
  // CHECK: @llvm.x86.avx512.mask.lzcnt.d.512
  return _mm512_mask_lzcnt_epi32(__W,__U,__A); 
}
__m512i test_mm512_maskz_lzcnt_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_lzcnt_epi32
  // CHECK: @llvm.x86.avx512.mask.lzcnt.d.512
  return _mm512_maskz_lzcnt_epi32(__U,__A); 
}
__m512i test_mm512_lzcnt_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_lzcnt_epi64
  // CHECK: @llvm.x86.avx512.mask.lzcnt.q.512
  return _mm512_lzcnt_epi64(__A); 
}
__m512i test_mm512_mask_lzcnt_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_lzcnt_epi64
  // CHECK: @llvm.x86.avx512.mask.lzcnt.q.512
  return _mm512_mask_lzcnt_epi64(__W,__U,__A); 
}
__m512i test_mm512_maskz_lzcnt_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_lzcnt_epi64
  // CHECK: @llvm.x86.avx512.mask.lzcnt.q.512
  return _mm512_maskz_lzcnt_epi64(__U,__A); 
}
