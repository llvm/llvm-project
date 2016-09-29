// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

__m512i test_mm512_mask2_permutex2var_epi8(__m512i __A, __m512i __I, __mmask64 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask2_permutex2var_epi8
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.qi.512
  return _mm512_mask2_permutex2var_epi8(__A, __I, __U, __B); 
}

__m512i test_mm512_permutex2var_epi8(__m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_permutex2var_epi8
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.qi.512
  return _mm512_permutex2var_epi8(__A, __I, __B); 
}

__m512i test_mm512_mask_permutex2var_epi8(__m512i __A, __mmask64 __U, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_permutex2var_epi8
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.qi.512
  return _mm512_mask_permutex2var_epi8(__A, __U, __I, __B); 
}

__m512i test_mm512_maskz_permutex2var_epi8(__mmask64 __U, __m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_permutex2var_epi8
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.qi.512
  return _mm512_maskz_permutex2var_epi8(__U, __A, __I, __B); 
}

__m512i test_mm512_permutexvar_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_permutexvar_epi8
  // CHECK: @llvm.x86.avx512.mask.permvar.qi.512
  return _mm512_permutexvar_epi8(__A, __B); 
}

__m512i test_mm512_maskz_permutexvar_epi8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_permutexvar_epi8
  // CHECK: @llvm.x86.avx512.mask.permvar.qi.512
  return _mm512_maskz_permutexvar_epi8(__M, __A, __B); 
}

__m512i test_mm512_mask_permutexvar_epi8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_permutexvar_epi8
  // CHECK: @llvm.x86.avx512.mask.permvar.qi.512
  return _mm512_mask_permutexvar_epi8(__W, __M, __A, __B); 
}

__m512i test_mm512_mask_multishift_epi64_epi8(__m512i __W, __mmask64 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_multishift_epi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmultishift.qb.512
  return _mm512_mask_multishift_epi64_epi8(__W, __M, __X, __Y); 
}

__m512i test_mm512_maskz_multishift_epi64_epi8(__mmask64 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maskz_multishift_epi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmultishift.qb.512
  return _mm512_maskz_multishift_epi64_epi8(__M, __X, __Y); 
}

__m512i test_mm512_multishift_epi64_epi8(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_multishift_epi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmultishift.qb.512
  return _mm512_multishift_epi64_epi8(__X, __Y); 
}
