// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bitalg -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512i test_mm512_popcnt_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_popcnt_epi16
  // CHECK: @llvm.ctpop.v32i16
  return _mm512_popcnt_epi16(__A);
}

__m512i test_mm512_mask_popcnt_epi16(__m512i __A, __mmask32 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_popcnt_epi16
  // CHECK: @llvm.ctpop.v32i16
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i16> %{{[0-9]+}}, <32 x i16> {{.*}}
  return _mm512_mask_popcnt_epi16(__A, __U, __B);
}
__m512i test_mm512_maskz_popcnt_epi16(__mmask32 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_popcnt_epi16
  // CHECK: @llvm.ctpop.v32i16
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i16> %{{[0-9]+}}, <32 x i16> {{.*}}
  return _mm512_maskz_popcnt_epi16(__U, __B);
}

__m512i test_mm512_popcnt_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_popcnt_epi8
  // CHECK: @llvm.ctpop.v64i8
  return _mm512_popcnt_epi8(__A);
}

__m512i test_mm512_mask_popcnt_epi8(__m512i __A, __mmask64 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_popcnt_epi8
  // CHECK: @llvm.ctpop.v64i8
  // CHECK: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_mask_popcnt_epi8(__A, __U, __B);
}
__m512i test_mm512_maskz_popcnt_epi8(__mmask64 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_popcnt_epi8
  // CHECK: @llvm.ctpop.v64i8
  // CHECK: select <64 x i1> %{{[0-9]+}}, <64 x i8> %{{[0-9]+}}, <64 x i8> {{.*}}
  return _mm512_maskz_popcnt_epi8(__U, __B);
}

__mmask64 test_mm512_mask_bitshuffle_epi64_mask(__mmask64 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_bitshuffle_epi64_mask
  // CHECK: @llvm.x86.avx512.vpshufbitqmb.512
  // CHECK: and <64 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_bitshuffle_epi64_mask(__U, __A, __B);
}

__mmask64 test_mm512_bitshuffle_epi64_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_bitshuffle_epi64_mask
  // CHECK: @llvm.x86.avx512.vpshufbitqmb.512
  return _mm512_bitshuffle_epi64_mask(__A, __B);
}

