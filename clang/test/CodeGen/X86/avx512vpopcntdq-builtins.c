// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vpopcntdq -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512i test_mm512_popcnt_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_popcnt_epi64
  // CHECK: @llvm.ctpop.v8i64
  return _mm512_popcnt_epi64(__A);
}
__m512i test_mm512_mask_popcnt_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_popcnt_epi64
  // CHECK: @llvm.ctpop.v8i64
  // CHECK: select <8 x i1> %{{[0-9]+}}, <8 x i64> %{{[0-9]+}}, <8 x i64> {{.*}}
  return _mm512_mask_popcnt_epi64(__W, __U, __A);
}
__m512i test_mm512_maskz_popcnt_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_popcnt_epi64
  // CHECK: @llvm.ctpop.v8i64
  // CHECK: select <8 x i1> %{{[0-9]+}}, <8 x i64> %{{[0-9]+}}, <8 x i64> {{.*}}
  return _mm512_maskz_popcnt_epi64(__U, __A);
}
__m512i test_mm512_popcnt_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_popcnt_epi32
  // CHECK: @llvm.ctpop.v16i32
  return _mm512_popcnt_epi32(__A);
}
__m512i test_mm512_mask_popcnt_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_popcnt_epi32
  // CHECK: @llvm.ctpop.v16i32
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, <16 x i32> {{.*}}
  return _mm512_mask_popcnt_epi32(__W, __U, __A);
}
__m512i test_mm512_maskz_popcnt_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_popcnt_epi32
  // CHECK: @llvm.ctpop.v16i32
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i32> %{{[0-9]+}}, <16 x i32> {{.*}}
  return _mm512_maskz_popcnt_epi32(__U, __A);
}
