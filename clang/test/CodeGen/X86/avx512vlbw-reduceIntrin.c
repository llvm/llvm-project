// RUN: %clang_cc1 -ffreestanding %s -O0 -triple=x86_64 -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -O0 -triple=i386 -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

short test_mm_reduce_add_epi16(__m128i __W){
// CHECK-LABEL: @test_mm_reduce_add_epi16(
// CHECK: call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_add_epi16(__W);
}

short test_mm_reduce_mul_epi16(__m128i __W){
// CHECK-LABEL: @test_mm_reduce_mul_epi16(
// CHECK:    call i16 @llvm.vector.reduce.mul.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_mul_epi16(__W);
}

short test_mm_reduce_or_epi16(__m128i __W){
// CHECK-LABEL: @test_mm_reduce_or_epi16(
// CHECK:    call i16 @llvm.vector.reduce.or.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_or_epi16(__W);
}

short test_mm_reduce_and_epi16(__m128i __W){
// CHECK-LABEL: @test_mm_reduce_and_epi16(
// CHECK:    call i16 @llvm.vector.reduce.and.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_and_epi16(__W);
}

short test_mm_mask_reduce_add_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: @test_mm_mask_reduce_add_epi16(
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_add_epi16(__M, __W);
}

short test_mm_mask_reduce_mul_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: @test_mm_mask_reduce_mul_epi16(
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.mul.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_mul_epi16(__M, __W);
}

short test_mm_mask_reduce_and_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: @test_mm_mask_reduce_and_epi16(
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.and.v8i16(<8 x i16> %{{.*}}
  return _mm_mask_reduce_and_epi16(__M, __W);
}

short test_mm_mask_reduce_or_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: @test_mm_mask_reduce_or_epi16(
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.or.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_or_epi16(__M, __W);
}

short test_mm256_reduce_add_epi16(__m256i __W){
// CHECK-LABEL: @test_mm256_reduce_add_epi16(
// CHECK:    call i16 @llvm.vector.reduce.add.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_add_epi16(__W);
}

short test_mm256_reduce_mul_epi16(__m256i __W){
// CHECK-LABEL: @test_mm256_reduce_mul_epi16(
// CHECK:    call i16 @llvm.vector.reduce.mul.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_mul_epi16(__W);
}

short test_mm256_reduce_or_epi16(__m256i __W){
// CHECK-LABEL: @test_mm256_reduce_or_epi16(
// CHECK:    call i16 @llvm.vector.reduce.or.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_or_epi16(__W);
}

short test_mm256_reduce_and_epi16(__m256i __W){
// CHECK-LABEL: @test_mm256_reduce_and_epi16(
// CHECK:    call i16 @llvm.vector.reduce.and.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_and_epi16(__W);
}

short test_mm256_mask_reduce_add_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: @test_mm256_mask_reduce_add_epi16(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.add.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_add_epi16(__M, __W);
}

short test_mm256_mask_reduce_mul_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: @test_mm256_mask_reduce_mul_epi16(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.mul.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_mul_epi16(__M, __W);
}

short test_mm256_mask_reduce_and_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: @test_mm256_mask_reduce_and_epi16(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.and.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_and_epi16(__M, __W);
}

short test_mm256_mask_reduce_or_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: @test_mm256_mask_reduce_or_epi16(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.or.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_or_epi16(__M, __W);
}

signed char test_mm_reduce_add_epi8(__m128i __W){
// CHECK-LABEL: @test_mm_reduce_add_epi8(
// CHECK:    call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_add_epi8(__W);
}

signed char test_mm_reduce_mul_epi8(__m128i __W){
// CHECK-LABEL: @test_mm_reduce_mul_epi8(
// CHECK:    call i8 @llvm.vector.reduce.mul.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_mul_epi8(__W);
}

signed char test_mm_reduce_and_epi8(__m128i __W){
// CHECK-LABEL: @test_mm_reduce_and_epi8(
// CHECK:    call i8 @llvm.vector.reduce.and.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_and_epi8(__W);
}

signed char test_mm_reduce_or_epi8(__m128i __W){
// CHECK-LABEL: @test_mm_reduce_or_epi8(
// CHECK:    call i8 @llvm.vector.reduce.or.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_or_epi8(__W);
}

signed char test_mm_mask_reduce_add_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: @test_mm_mask_reduce_add_epi8(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_add_epi8(__M, __W);
}

signed char test_mm_mask_reduce_mul_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: @test_mm_mask_reduce_mul_epi8(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.mul.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_mul_epi8(__M, __W);
}

signed char test_mm_mask_reduce_and_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: @test_mm_mask_reduce_and_epi8(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.and.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_and_epi8(__M, __W);
}

signed char test_mm_mask_reduce_or_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: @test_mm_mask_reduce_or_epi8(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.or.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_or_epi8(__M, __W);
}

signed char test_mm256_reduce_add_epi8(__m256i __W){
// CHECK-LABEL: @test_mm256_reduce_add_epi8(
// CHECK:    call i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_add_epi8(__W);
}

signed char test_mm256_reduce_mul_epi8(__m256i __W){
// CHECK-LABEL: @test_mm256_reduce_mul_epi8(
// CHECK:    call i8 @llvm.vector.reduce.mul.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_mul_epi8(__W);
}

signed char test_mm256_reduce_and_epi8(__m256i __W){
// CHECK-LABEL: @test_mm256_reduce_and_epi8(
// CHECK:    call i8 @llvm.vector.reduce.and.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_and_epi8(__W);
}

signed char test_mm256_reduce_or_epi8(__m256i __W){
// CHECK-LABEL: @test_mm256_reduce_or_epi8(
// CHECK:    call i8 @llvm.vector.reduce.or.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_or_epi8(__W);
}

signed char test_mm256_mask_reduce_add_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: @test_mm256_mask_reduce_add_epi8(
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_add_epi8(__M, __W);
}

signed char test_mm256_mask_reduce_mul_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: @test_mm256_mask_reduce_mul_epi8(
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.mul.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_mul_epi8(__M, __W);
}

signed char test_mm256_mask_reduce_and_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: @test_mm256_mask_reduce_and_epi8(
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.and.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_and_epi8(__M, __W);
}

signed char test_mm256_mask_reduce_or_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: @test_mm256_mask_reduce_or_epi8(
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.or.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_or_epi8(__M, __W);
}

short test_mm_reduce_max_epi16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_max_epi16
// CHECK:    call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_max_epi16(__W);
}

short test_mm_reduce_min_epi16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_min_epi16
// CHECK:    call i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_min_epi16(__W);
}

unsigned short test_mm_reduce_max_epu16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_max_epu16
// CHECK:    call i16 @llvm.vector.reduce.umax.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_max_epu16(__W);
}

unsigned short test_mm_reduce_min_epu16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_min_epu16
// CHECK:    call i16 @llvm.vector.reduce.umin.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_min_epu16(__W);
}

short test_mm_mask_reduce_max_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_max_epi16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_max_epi16(__M, __W);
}

short test_mm_mask_reduce_min_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_min_epi16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_min_epi16(__M, __W);
}

unsigned short test_mm_mask_reduce_max_epu16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_max_epu16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.umax.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_max_epu16(__M, __W);
}

unsigned short test_mm_mask_reduce_min_epu16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_min_epu16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.umin.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_min_epu16(__M, __W);
}

short test_mm256_reduce_max_epi16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_max_epi16
// CHECK:    call i16 @llvm.vector.reduce.smax.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_max_epi16(__W);
}

short test_mm256_reduce_min_epi16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_min_epi16
// CHECK:    call i16 @llvm.vector.reduce.smin.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_min_epi16(__W);
}

unsigned short test_mm256_reduce_max_epu16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_max_epu16
// CHECK:    call i16 @llvm.vector.reduce.umax.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_max_epu16(__W);
}

unsigned short test_mm256_reduce_min_epu16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_min_epu16
// CHECK:    call i16 @llvm.vector.reduce.umin.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_min_epu16(__W);
}

short test_mm256_mask_reduce_max_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_max_epi16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.smax.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_max_epi16(__M, __W);
}

short test_mm256_mask_reduce_min_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_min_epi16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.smin.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_min_epi16(__M, __W);
}

unsigned short test_mm256_mask_reduce_max_epu16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_max_epu16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.umax.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_max_epu16(__M, __W);
}

unsigned short test_mm256_mask_reduce_min_epu16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_min_epu16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call i16 @llvm.vector.reduce.umin.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_min_epu16(__M, __W);
}

signed char test_mm_reduce_max_epi8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_max_epi8
// CHECK:    call i8 @llvm.vector.reduce.smax.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_max_epi8(__W);
}

signed char test_mm_reduce_min_epi8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_min_epi8
// CHECK:    call i8 @llvm.vector.reduce.smin.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_min_epi8(__W);
}

unsigned char test_mm_reduce_max_epu8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_max_epu8
// CHECK:    call i8 @llvm.vector.reduce.umax.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_max_epu8(__W);
}

unsigned char test_mm_reduce_min_epu8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_min_epu8
// CHECK:    call i8 @llvm.vector.reduce.umin.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_min_epu8(__W);
}

signed char test_mm_mask_reduce_max_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_max_epi8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.smax.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_max_epi8(__M, __W);
}

signed char test_mm_mask_reduce_min_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_min_epi8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.smin.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_min_epi8(__M, __W);
}

unsigned char test_mm_mask_reduce_max_epu8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_max_epu8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.umax.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_max_epu8(__M, __W);
}

unsigned char test_mm_mask_reduce_min_epu8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_min_epu8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.umin.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_min_epu8(__M, __W);
}

signed char test_mm256_reduce_max_epi8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_max_epi8
// CHECK:    call i8 @llvm.vector.reduce.smax.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_max_epi8(__W);
}

signed char test_mm256_reduce_min_epi8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_min_epi8
// CHECK:    call i8 @llvm.vector.reduce.smin.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_min_epi8(__W);
}

unsigned char test_mm256_reduce_max_epu8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_max_epu8
// CHECK:    call i8 @llvm.vector.reduce.umax.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_max_epu8(__W);
}

unsigned char test_mm256_reduce_min_epu8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_min_epu8
// CHECK:    call i8 @llvm.vector.reduce.umin.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_min_epu8(__W);
}

signed char test_mm256_mask_reduce_max_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_max_epi8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.smax.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_max_epi8(__M, __W);
}

signed char test_mm256_mask_reduce_min_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_min_epi8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.smin.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_min_epi8(__M, __W);
}

unsigned char test_mm256_mask_reduce_max_epu8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_max_epu8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.umax.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_max_epu8(__M, __W);
}

unsigned char test_mm256_mask_reduce_min_epu8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_min_epu8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call i8 @llvm.vector.reduce.umin.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_min_epu8(__M, __W);
}
