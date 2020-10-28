// RUN: %clang_cc1 -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

long long test_mm512_reduce_add_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_add_epi64(
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    add <4 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    add <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    add <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x i64> %{{.*}}, i32 0
  return _mm512_reduce_add_epi64(__W);
}

long long test_mm512_reduce_mul_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_mul_epi64(
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    mul <4 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    mul <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    mul <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x i64> %{{.*}}, i32 0
  return _mm512_reduce_mul_epi64(__W); 
}

long long test_mm512_reduce_or_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_or_epi64(
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    or <4 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    or <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    or <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x i64> %{{.*}}, i32 0
  return _mm512_reduce_or_epi64(__W); 
}

long long test_mm512_reduce_and_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_and_epi64(
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    and <4 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    and <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    and <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x i64> %{{.*}}, i32 0
  return _mm512_reduce_and_epi64(__W);
}

long long test_mm512_mask_reduce_add_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    add <4 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    add <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    add <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x i64> %{{.*}}, i32 0
  return _mm512_mask_reduce_add_epi64(__M, __W); 
}

long long test_mm512_mask_reduce_mul_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    mul <4 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    mul <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    mul <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x i64> %{{.*}}, i32 0
  return _mm512_mask_reduce_mul_epi64(__M, __W); 
}

long long test_mm512_mask_reduce_and_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_and_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    and <4 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    and <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    and <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x i64> %{{.*}}, i32 0
  return _mm512_mask_reduce_and_epi64(__M, __W); 
}

long long test_mm512_mask_reduce_or_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_or_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    or <4 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    or <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    or <2 x i64> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x i64> %{{.*}}, i32 0
  return _mm512_mask_reduce_or_epi64(__M, __W); 
}

int test_mm512_reduce_add_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_add_epi32(
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    add <8 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    add <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    add <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    add <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x i32> %{{.*}}, i32 0
  return _mm512_reduce_add_epi32(__W); 
}

int test_mm512_reduce_mul_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_mul_epi32(
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    mul <8 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    mul <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    mul <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    mul <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x i32> %{{.*}}, i32 0
  return _mm512_reduce_mul_epi32(__W); 
}

int test_mm512_reduce_or_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_or_epi32(
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    or <8 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    or <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    or <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    or <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x i32> %{{.*}}, i32 0
  return _mm512_reduce_or_epi32(__W); 
}

int test_mm512_reduce_and_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_and_epi32(
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    and <8 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    and <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    and <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    and <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x i32> %{{.*}}, i32 0
  return _mm512_reduce_and_epi32(__W); 
}

int test_mm512_mask_reduce_add_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    bitcast <16 x i32> %{{.*}} to <8 x i64>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    add <8 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    add <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    add <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    add <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x i32> %{{.*}}, i32 0
  return _mm512_mask_reduce_add_epi32(__M, __W); 
}

int test_mm512_mask_reduce_mul_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    bitcast <16 x i32> %{{.*}} to <8 x i64>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    mul <8 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    mul <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    mul <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    mul <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x i32> %{{.*}}, i32 0
  return _mm512_mask_reduce_mul_epi32(__M, __W); 
}

int test_mm512_mask_reduce_and_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_and_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    bitcast <16 x i32> %{{.*}} to <8 x i64>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    and <8 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    and <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    and <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    and <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x i32> %{{.*}}, i32 0
  return _mm512_mask_reduce_and_epi32(__M, __W); 
}

int test_mm512_mask_reduce_or_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_or_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    bitcast <16 x i32> %{{.*}} to <8 x i64>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x i64> %{{.*}}, <8 x i64> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    or <8 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    or <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    or <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    or <4 x i32> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x i32> %{{.*}}, i32 0
  return _mm512_mask_reduce_or_epi32(__M, __W); 
}

double test_mm512_reduce_add_pd(__m512d __W){
// CHECK-LABEL: @test_mm512_reduce_add_pd(
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    fadd <4 x double> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    fadd <2 x double> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    fadd <2 x double> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x double> %{{.*}}, i32 0
  return _mm512_reduce_add_pd(__W); 
}

double test_mm512_reduce_mul_pd(__m512d __W){
// CHECK-LABEL: @test_mm512_reduce_mul_pd(
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    fmul <4 x double> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    fmul <2 x double> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    fmul <2 x double> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x double> %{{.*}}, i32 0
  return _mm512_reduce_mul_pd(__W); 
}

float test_mm512_reduce_add_ps(__m512 __W){
// CHECK-LABEL: @test_mm512_reduce_add_ps(
// CHECK:    bitcast <16 x float> %{{.*}} to <8 x double>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    bitcast <4 x double> %{{.*}} to <8 x float>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    bitcast <4 x double> %{{.*}} to <8 x float>
// CHECK:    fadd <8 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    fadd <4 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    fadd <4 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    fadd <4 x float> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x float> %{{.*}}, i32 0
  return _mm512_reduce_add_ps(__W); 
}

float test_mm512_reduce_mul_ps(__m512 __W){
// CHECK-LABEL: @test_mm512_reduce_mul_ps(
// CHECK:    bitcast <16 x float> %{{.*}} to <8 x double>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    bitcast <4 x double> %{{.*}} to <8 x float>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    bitcast <4 x double> %{{.*}} to <8 x float>
// CHECK:    fmul <8 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    fmul <4 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    fmul <4 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    fmul <4 x float> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x float> %{{.*}}, i32 0
  return _mm512_reduce_mul_ps(__W); 
}

double test_mm512_mask_reduce_add_pd(__mmask8 __M, __m512d __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_pd(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    fadd <4 x double> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    fadd <2 x double> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    fadd <2 x double> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x double> %{{.*}}, i32 0
  return _mm512_mask_reduce_add_pd(__M, __W); 
}

double test_mm512_mask_reduce_mul_pd(__mmask8 __M, __m512d __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_pd(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    fmul <4 x double> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    fmul <2 x double> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    fmul <2 x double> %{{.*}}, %{{.*}}
// CHECK:    extractelement <2 x double> %{{.*}}, i32 0
  return _mm512_mask_reduce_mul_pd(__M, __W); 
}

float test_mm512_mask_reduce_add_ps(__mmask16 __M, __m512 __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_ps(
// CHECK-NEXT:  entry:
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> {{.*}}, <16 x float> {{.*}}
// CHECK:    bitcast <16 x float> %{{.*}} to <8 x double>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    bitcast <4 x double> %{{.*}} to <8 x float>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    bitcast <4 x double> %{{.*}} to <8 x float>
// CHECK:    fadd <8 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    fadd <4 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    fadd <4 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    fadd <4 x float> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x float> %{{.*}}, i32 0
  return _mm512_mask_reduce_add_ps(__M, __W); 
}

float test_mm512_mask_reduce_mul_ps(__mmask16 __M, __m512 __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_ps(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> {{.*}}, <16 x float> %{{.*}}
// CHECK:    bitcast <16 x float> %{{.*}} to <8 x double>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    bitcast <4 x double> %{{.*}} to <8 x float>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    bitcast <4 x double> %{{.*}} to <8 x float>
// CHECK:    fmul <8 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    fmul <4 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    fmul <4 x float> %{{.*}}, %{{.*}}
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    fmul <4 x float> %{{.*}}, %{{.*}}
// CHECK:    extractelement <4 x float> %{{.*}}, i32 0
  return _mm512_mask_reduce_mul_ps(__M, __W); 
}
