// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512pf -emit-llvm -o - -Wall -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

void test_mm512_mask_prefetch_i32gather_pd(__m256i index, __mmask8 mask, void const *addr, int hint) {
  // CHECK-LABEL: @test_mm512_mask_prefetch_i32gather_pd
  // CHECK: @llvm.x86.avx512.gatherpf.dpd
  return _mm512_mask_prefetch_i32gather_pd(index, mask, addr, 2, 1); 
}

void test_mm512_prefetch_i32gather_pd(__m256i index, void const *addr, int hint) {
  // CHECK-LABEL: @test_mm512_prefetch_i32gather_pd
  // CHECK: @llvm.x86.avx512.gatherpf.dpd
  return _mm512_prefetch_i32gather_pd(index, addr, 2, 1); 
}

void test_mm512_mask_prefetch_i32gather_ps(__m512i index, __mmask16 mask, void const *addr, int hint) {
  // CHECK-LABEL: @test_mm512_mask_prefetch_i32gather_ps
  // CHECK: @llvm.x86.avx512.gatherpf.dps
  return _mm512_mask_prefetch_i32gather_ps(index, mask, addr, 2, 1); 
}

void test_mm512_prefetch_i32gather_ps(__m512i index,  void const *addr, int hint) {
  // CHECK-LABEL: @test_mm512_prefetch_i32gather_ps
  // CHECK: @llvm.x86.avx512.gatherpf.dps
  return _mm512_prefetch_i32gather_ps(index, addr, 2, 1); 
}

void test_mm512_mask_prefetch_i64gather_pd(__m512i index, __mmask8 mask, void const *addr, int hint) {
  // CHECK-LABEL: @test_mm512_mask_prefetch_i64gather_pd
  // CHECK: @llvm.x86.avx512.gatherpf.qpd
  return _mm512_mask_prefetch_i64gather_pd(index, mask, addr, 2, 1); 
}

void test_mm512_prefetch_i64gather_pd(__m512i index, void const *addr, int hint) {
  // CHECK-LABEL: @test_mm512_prefetch_i64gather_pd
  // CHECK: @llvm.x86.avx512.gatherpf.qpd
  return _mm512_prefetch_i64gather_pd(index, addr, 2, 1); 
}

void test_mm512_mask_prefetch_i64gather_ps(__m512i index, __mmask8 mask, void const *addr, int hint) {
  // CHECK-LABEL: @test_mm512_mask_prefetch_i64gather_ps
  // CHECK: @llvm.x86.avx512.gatherpf.qps
  return _mm512_mask_prefetch_i64gather_ps(index, mask, addr, 2, 1); 
}

void test_mm512_prefetch_i64gather_ps(__m512i index, void const *addr, int hint) {
  // CHECK-LABEL: @test_mm512_prefetch_i64gather_ps
  // CHECK: @llvm.x86.avx512.gatherpf.qps
  return _mm512_prefetch_i64gather_ps(index, addr, 2, 1); 
}

void test_mm512_prefetch_i32scatter_pd(void *addr, __m256i index) {
  // CHECK-LABEL: @test_mm512_prefetch_i32scatter_pd
  // CHECK: @llvm.x86.avx512.scatterpf.dpd.512
  return _mm512_prefetch_i32scatter_pd(addr, index, 1, 2); 
}

void test_mm512_mask_prefetch_i32scatter_pd(void *addr, __mmask8 mask, __m256i index) {
  // CHECK-LABEL: @test_mm512_mask_prefetch_i32scatter_pd
  // CHECK: @llvm.x86.avx512.scatterpf.dpd.512
  return _mm512_mask_prefetch_i32scatter_pd(addr, mask, index, 1, 2); 
}

void test_mm512_prefetch_i32scatter_ps(void *addr, __m512i index) {
  // CHECK-LABEL: @test_mm512_prefetch_i32scatter_ps
  // CHECK: @llvm.x86.avx512.scatterpf.dps.512
  return _mm512_prefetch_i32scatter_ps(addr, index, 1, 2); 
}

void test_mm512_mask_prefetch_i32scatter_ps(void *addr, __mmask16 mask, __m512i index) {
  // CHECK-LABEL: @test_mm512_mask_prefetch_i32scatter_ps
  // CHECK: @llvm.x86.avx512.scatterpf.dps.512
  return _mm512_mask_prefetch_i32scatter_ps(addr, mask, index, 1, 2); 
}

void test_mm512_prefetch_i64scatter_pd(void *addr, __m512i index) {
  // CHECK-LABEL: @test_mm512_prefetch_i64scatter_pd
  // CHECK: @llvm.x86.avx512.scatterpf.qpd.512
  return _mm512_prefetch_i64scatter_pd(addr, index, 1, 2); 
}

void test_mm512_mask_prefetch_i64scatter_pd(void *addr, __mmask16 mask, __m512i index) {
  // CHECK-LABEL: @test_mm512_mask_prefetch_i64scatter_pd
  // CHECK: @llvm.x86.avx512.scatterpf.qpd.512
  return _mm512_mask_prefetch_i64scatter_pd(addr, mask, index, 1, 2); 
}

void test_mm512_prefetch_i64scatter_ps(void *addr, __m512i index) {
  // CHECK-LABEL: @test_mm512_prefetch_i64scatter_ps
  // CHECK: @llvm.x86.avx512.scatterpf.qps.512
  return _mm512_prefetch_i64scatter_ps(addr, index, 1, 2); 
}

void test_mm512_mask_prefetch_i64scatter_ps(void *addr, __mmask16 mask, __m512i index) {
  // CHECK-LABEL: @test_mm512_mask_prefetch_i64scatter_ps
  // CHECK: @llvm.x86.avx512.scatterpf.qps.512
  return _mm512_mask_prefetch_i64scatter_ps(addr, mask, index, 1, 2); 
}
