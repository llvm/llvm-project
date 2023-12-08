// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512ifma -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

__m512i test_mm512_madd52hi_epu64(__m512i __X, __m512i __Y, __m512i __Z) {
  // CHECK-LABEL: @test_mm512_madd52hi_epu64
  // CHECK: @llvm.x86.avx512.vpmadd52h.uq.512
  return _mm512_madd52hi_epu64(__X, __Y, __Z); 
}

__m512i test_mm512_mask_madd52hi_epu64(__m512i __W, __mmask8 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_madd52hi_epu64
  // CHECK: @llvm.x86.avx512.vpmadd52h.uq.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_madd52hi_epu64(__W, __M, __X, __Y); 
}

__m512i test_mm512_maskz_madd52hi_epu64(__mmask8 __M, __m512i __X, __m512i __Y, __m512i __Z) {
  // CHECK-LABEL: @test_mm512_maskz_madd52hi_epu64
  // CHECK: @llvm.x86.avx512.vpmadd52h.uq.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_madd52hi_epu64(__M, __X, __Y, __Z); 
}

__m512i test_mm512_madd52lo_epu64(__m512i __X, __m512i __Y, __m512i __Z) {
  // CHECK-LABEL: @test_mm512_madd52lo_epu64
  // CHECK: @llvm.x86.avx512.vpmadd52l.uq.512
  return _mm512_madd52lo_epu64(__X, __Y, __Z); 
}

__m512i test_mm512_mask_madd52lo_epu64(__m512i __W, __mmask8 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_madd52lo_epu64
  // CHECK: @llvm.x86.avx512.vpmadd52l.uq.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_madd52lo_epu64(__W, __M, __X, __Y); 
}

__m512i test_mm512_maskz_madd52lo_epu64(__mmask8 __M, __m512i __X, __m512i __Y, __m512i __Z) {
  // CHECK-LABEL: @test_mm512_maskz_madd52lo_epu64
  // CHECK: @llvm.x86.avx512.vpmadd52l.uq.512
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_madd52lo_epu64(__M, __X, __Y, __Z); 
}
