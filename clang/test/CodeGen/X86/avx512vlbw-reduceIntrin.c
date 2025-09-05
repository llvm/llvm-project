// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64 -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=x86_64 -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=i386 -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -O0 -triple=i386 -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -O0 -triple=x86_64 -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -O0 -triple=x86_64 -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -O0 -triple=i386 -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -O0 -triple=i386 -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

short test_mm_reduce_add_epi16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_add_epi16
// CHECK: call {{.*}}i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_add_epi16(__W);
}
TEST_CONSTEXPR(_mm_reduce_add_epi16((__m128i)(__v8hi){1,2,3,4,5,6,7,8}) == 36);

short test_mm_reduce_mul_epi16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_mul_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.mul.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_mul_epi16(__W);
}
TEST_CONSTEXPR(_mm_reduce_mul_epi16((__m128i)(__v8hi){1,2,3,1,2,3,1,2}) == 72);

short test_mm_reduce_or_epi16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_or_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.or.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_or_epi16(__W);
}
TEST_CONSTEXPR(_mm_reduce_or_epi16((__m128i)(__v8hi){1,2,4,8,0,0,0,0}) == 15);

short test_mm_reduce_and_epi16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_and_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.and.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_and_epi16(__W);
}
TEST_CONSTEXPR(_mm_reduce_and_epi16((__m128i)(__v8hi){1,3,5,7,9,11,13,15}) == 1);

short test_mm_mask_reduce_add_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_add_epi16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_add_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_add_epi16((__mmask8)0b11110000, (__m128i)(__v8hi){1,2,3,4,5,6,7,8}) == 26);
TEST_CONSTEXPR(_mm_mask_reduce_add_epi16((__mmask8)0b00001111, (__m128i)(__v8hi){1,2,3,4,5,6,7,8}) == 10);

short test_mm_mask_reduce_mul_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_mul_epi16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.mul.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_mul_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_mul_epi16((__mmask8)0b11110000, (__m128i)(__v8hi){1,2,3,1,2,3,1,2}) == 12);
TEST_CONSTEXPR(_mm_mask_reduce_mul_epi16((__mmask8)0b00001111, (__m128i)(__v8hi){1,2,3,1,2,3,1,2}) == 6);

short test_mm_mask_reduce_and_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_and_epi16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.and.v8i16(<8 x i16> %{{.*}}
  return _mm_mask_reduce_and_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_and_epi16((__mmask8)0b11110000, (__m128i)(__v8hi){1,3,5,7,0,0,0,0}) == 0);
TEST_CONSTEXPR(_mm_mask_reduce_and_epi16((__mmask8)0b00001111, (__m128i)(__v8hi){1,3,5,7,0,0,0,0}) == 1);

short test_mm_mask_reduce_or_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_or_epi16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.or.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_or_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_or_epi16((__mmask8)0b11110000, (__m128i)(__v8hi){1,2,4,8,0,0,0,0}) == 0);
TEST_CONSTEXPR(_mm_mask_reduce_or_epi16((__mmask8)0b00001111, (__m128i)(__v8hi){1,2,4,8,0,0,0,0}) == 15);

short test_mm256_reduce_add_epi16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_add_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.add.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_add_epi16(__W);
}
TEST_CONSTEXPR(_mm256_reduce_add_epi16((__m256i)(__v16hi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) == 120);

short test_mm256_reduce_mul_epi16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_mul_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.mul.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_mul_epi16(__W);
}
TEST_CONSTEXPR(_mm256_reduce_mul_epi16((__m256i)(__v16hi){1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1}) == 7776);

short test_mm256_reduce_or_epi16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_or_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.or.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_or_epi16(__W);
}
TEST_CONSTEXPR(_mm256_reduce_or_epi16((__m256i)(__v16hi){1,2,4,8,16,32,64,128,0,0,0,0,0,0,0,0}) == 255);

short test_mm256_reduce_and_epi16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_and_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.and.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_and_epi16(__W);
}
TEST_CONSTEXPR(_mm256_reduce_and_epi16((__m256i)(__v16hi){1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31}) == 1);

short test_mm256_mask_reduce_add_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_add_epi16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.add.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_add_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_add_epi16((__mmask16)0b1111111100000000, (__m256i)(__v16hi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) == 84);
TEST_CONSTEXPR(_mm256_mask_reduce_add_epi16((__mmask16)0b0000000011111111, (__m256i)(__v16hi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) == 36);

short test_mm256_mask_reduce_mul_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_mul_epi16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.mul.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_mul_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_mul_epi16((__mmask16)0b1111111100000000, (__m256i)(__v16hi){1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1}) == 108);
TEST_CONSTEXPR(_mm256_mask_reduce_mul_epi16((__mmask16)0b0000000011111111, (__m256i)(__v16hi){1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1}) == 72);

short test_mm256_mask_reduce_and_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_and_epi16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.and.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_and_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_and_epi16((__mmask16)0b1111111100000000, (__m256i)(__v16hi){1,3,5,7,9,11,13,15,0,0,0,0,0,0,0,0}) == 0);
TEST_CONSTEXPR(_mm256_mask_reduce_and_epi16((__mmask16)0b0000000011111111, (__m256i)(__v16hi){1,3,5,7,9,11,13,15,0,0,0,0,0,0,0,0}) == 1);

short test_mm256_mask_reduce_or_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_or_epi16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.or.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_or_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_or_epi16((__mmask16)0b1111111100000000, (__m256i)(__v16hi){1,2,4,8,16,32,64,128,0,0,0,0,0,0,0,0}) == 0);
TEST_CONSTEXPR(_mm256_mask_reduce_or_epi16((__mmask16)0b0000000011111111, (__m256i)(__v16hi){1,2,4,8,16,32,64,128,0,0,0,0,0,0,0,0}) == 255);

signed char test_mm_reduce_add_epi8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_add_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_add_epi8(__W);
}
TEST_CONSTEXPR(_mm_reduce_add_epi8((__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) == 120);

signed char test_mm_reduce_mul_epi8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_mul_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.mul.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_mul_epi8(__W);
}
TEST_CONSTEXPR(_mm_reduce_mul_epi8((__m128i)(__v16qs){1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1}) == 32);

signed char test_mm_reduce_and_epi8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_and_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.and.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_and_epi8(__W);
}
TEST_CONSTEXPR(_mm_reduce_and_epi8((__m128i)(__v16qs){1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31}) == 1);

signed char test_mm_reduce_or_epi8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_or_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.or.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_or_epi8(__W);
}
TEST_CONSTEXPR(_mm_reduce_or_epi8((__m128i)(__v16qs){0,1,2,4,8,16,32,64,0,0,0,0,0,0,0,0}) == 127);

signed char test_mm_mask_reduce_add_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_add_epi8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_add_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_add_epi8((__mmask16)0b1111111100000000, (__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) == 84);
TEST_CONSTEXPR(_mm_mask_reduce_add_epi8((__mmask16)0b0000000011111111, (__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) == 36);

signed char test_mm_mask_reduce_mul_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_mul_epi8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.mul.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_mul_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_mul_epi8((__mmask16)0b1111111100000000, (__m128i)(__v16qs){1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1}) == 4);
TEST_CONSTEXPR(_mm_mask_reduce_mul_epi8((__mmask16)0b0000000011111111, (__m128i)(__v16qs){1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1}) == 8);

signed char test_mm_mask_reduce_and_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_and_epi8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.and.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_and_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_and_epi8((__mmask16)0b1111111100000000, (__m128i)(__v16qs){1,3,5,7,9,11,13,15,0,0,0,0,0,0,0,0}) == 0);
TEST_CONSTEXPR(_mm_mask_reduce_and_epi8((__mmask16)0b0000000011111111, (__m128i)(__v16qs){1,3,5,7,9,11,13,15,0,0,0,0,0,0,0,0}) == 1);

signed char test_mm_mask_reduce_or_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_or_epi8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.or.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_or_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_or_epi8((__mmask16)0b1111111100000000, (__m128i)(__v16qs){0,1,2,4,8,16,32,64,0,0,0,0,0,0,0,0}) == 0);
TEST_CONSTEXPR(_mm_mask_reduce_or_epi8((__mmask16)0b0000000011111111, (__m128i)(__v16qs){0,1,2,4,8,16,32,64,0,0,0,0,0,0,0,0}) == 127);

signed char test_mm256_reduce_add_epi8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_add_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_add_epi8(__W);
}
TEST_CONSTEXPR(_mm256_reduce_add_epi8((__m256i)(__v32qs){0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}) == 112);

signed char test_mm256_reduce_mul_epi8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_mul_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.mul.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_mul_epi8(__W);
}
TEST_CONSTEXPR(_mm256_reduce_mul_epi8((__m256i)(__v32qs){1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2}) == 16);

signed char test_mm256_reduce_and_epi8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_and_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.and.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_and_epi8(__W);
}
TEST_CONSTEXPR(_mm256_reduce_and_epi8((__m256i)(__v32qs){1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63}) == 1);

signed char test_mm256_reduce_or_epi8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_or_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.or.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_or_epi8(__W);
}
TEST_CONSTEXPR(_mm256_reduce_or_epi8((__m256i)(__v32qs){1,2,4,8,16,32,64,127,1,2,4,8,16,32,64,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}) == 127);

signed char test_mm256_mask_reduce_add_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_add_epi8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_add_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_add_epi8((__mmask32)0b11111111111111110000000000000000, (__m256i)(__v32qs){0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}) == 56);
TEST_CONSTEXPR(_mm256_mask_reduce_add_epi8((__mmask32)0b00000000000000001111111111111111, (__m256i)(__v32qs){8,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}) == 64);

signed char test_mm256_mask_reduce_mul_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_mul_epi8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.mul.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_mul_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_mul_epi8((__mmask32)0b11111111111111110000000000000000, (__m256i)(__v32qs){1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2}) == 4);
TEST_CONSTEXPR(_mm256_mask_reduce_mul_epi8((__mmask32)0b00000000000000001111111111111111, (__m256i)(__v32qs){4,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2}) == 16);

signed char test_mm256_mask_reduce_and_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_and_epi8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.and.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_and_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_and_epi8((__mmask32)0b11111111111111110000000000000000, (__m256i)(__v32qs){1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}) == 0);
TEST_CONSTEXPR(_mm256_mask_reduce_and_epi8((__mmask32)0b00000000000000001111111111111111, (__m256i)(__v32qs){1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}) == 1);

signed char test_mm256_mask_reduce_or_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_or_epi8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.or.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_or_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_or_epi8((__mmask32)0b11111111111111110000000000000000, (__m256i)(__v32qs){1,2,4,8,16,32,64,127,1,2,4,8,16,32,64,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}) == 0);
TEST_CONSTEXPR(_mm256_mask_reduce_or_epi8((__mmask32)0b00000000000000001111111111111111, (__m256i)(__v32qs){1,2,4,8,16,32,64,127,1,2,4,8,16,32,64,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}) == 127);

short test_mm_reduce_max_epi16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_max_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_max_epi16(__W);
}
TEST_CONSTEXPR(_mm_reduce_max_epi16((__m128i)(__v8hi){-4,-3,-2,-1,1,2,3,4}) == 4);

short test_mm_reduce_min_epi16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_min_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_min_epi16(__W);
}
TEST_CONSTEXPR(_mm_reduce_min_epi16((__m128i)(__v8hi){-4,-3,-2,-1,1,2,3,4}) == -4);

unsigned short test_mm_reduce_max_epu16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_max_epu16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.umax.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_max_epu16(__W);
}
TEST_CONSTEXPR(_mm_reduce_max_epu16((__m128i)(__v8hu){1,2,3,4,5,6,7,8}) == 8);

unsigned short test_mm_reduce_min_epu16(__m128i __W){
// CHECK-LABEL: test_mm_reduce_min_epu16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.umin.v8i16(<8 x i16> %{{.*}})
  return _mm_reduce_min_epu16(__W);
}
TEST_CONSTEXPR(_mm_reduce_min_epu16((__m128i)(__v8hu){1,2,3,4,5,6,7,8}) == 1);

short test_mm_mask_reduce_max_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_max_epi16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_max_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_max_epi16((__mmask8)0b11110000, (__m128i)(__v8hi){-4,-3,-2,-1,1,2,3,4}) == 4);
TEST_CONSTEXPR(_mm_mask_reduce_max_epi16((__mmask8)0b00001111, (__m128i)(__v8hi){-4,-3,-2,-1,1,2,3,4}) == -1);

short test_mm_mask_reduce_min_epi16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_min_epi16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.smin.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_min_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_min_epi16((__mmask8)0b11110000, (__m128i)(__v8hi){-4,-3,-2,-1,1,2,3,4}) == 1);
TEST_CONSTEXPR(_mm_mask_reduce_min_epi16((__mmask8)0b00001111, (__m128i)(__v8hi){-4,-3,-2,-1,1,2,3,4}) == -4);

unsigned short test_mm_mask_reduce_max_epu16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_max_epu16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.umax.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_max_epu16(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_max_epu16((__mmask8)0b11110000, (__m128i)(__v8hu){1,2,3,4,5,6,7,8}) == 8);
TEST_CONSTEXPR(_mm_mask_reduce_max_epu16((__mmask8)0b00001111, (__m128i)(__v8hu){1,2,3,4,5,6,7,8}) == 4);

unsigned short test_mm_mask_reduce_min_epu16(__mmask8 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_min_epu16
// CHECK:    select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.umin.v8i16(<8 x i16> %{{.*}})
  return _mm_mask_reduce_min_epu16(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_min_epu16((__mmask8)0b11110000, (__m128i)(__v8hu){1,2,3,4,5,6,7,8}) == 5);
TEST_CONSTEXPR(_mm_mask_reduce_min_epu16((__mmask8)0b00001111, (__m128i)(__v8hu){1,2,3,4,5,6,7,8}) == 1);

short test_mm256_reduce_max_epi16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_max_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.smax.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_max_epi16(__W);
}
TEST_CONSTEXPR(_mm256_reduce_max_epi16((__m256i)(__v16hi){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == 8);

short test_mm256_reduce_min_epi16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_min_epi16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.smin.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_min_epi16(__W);
}
TEST_CONSTEXPR(_mm256_reduce_min_epi16((__m256i)(__v16hi){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == -8);

unsigned short test_mm256_reduce_max_epu16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_max_epu16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.umax.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_max_epu16(__W);
}
TEST_CONSTEXPR(_mm256_reduce_max_epu16((__m256i)(__v16hu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 16);

unsigned short test_mm256_reduce_min_epu16(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_min_epu16
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.umin.v16i16(<16 x i16> %{{.*}})
  return _mm256_reduce_min_epu16(__W);
}
TEST_CONSTEXPR(_mm256_reduce_min_epu16((__m256i)(__v16hu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 1);

short test_mm256_mask_reduce_max_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_max_epi16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.smax.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_max_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_max_epi16((__mmask16){0b1111111100000000}, (__m256i)(__v16hi){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == 8);
TEST_CONSTEXPR(_mm256_mask_reduce_max_epi16((__mmask16){0b0000000011111111}, (__m256i)(__v16hi){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == -1);

short test_mm256_mask_reduce_min_epi16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_min_epi16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.smin.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_min_epi16(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_min_epi16((__mmask16){0b1111111100000000}, (__m256i)(__v16hi){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == 1);
TEST_CONSTEXPR(_mm256_mask_reduce_min_epi16((__mmask16){0b0000000011111111}, (__m256i)(__v16hi){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == -8);

unsigned short test_mm256_mask_reduce_max_epu16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_max_epu16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.umax.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_max_epu16(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_max_epu16((__mmask16){0b1111111100000000}, (__m256i)(__v16hu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 16);
TEST_CONSTEXPR(_mm256_mask_reduce_max_epu16((__mmask16){0b0000000011111111}, (__m256i)(__v16hu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 8);

unsigned short test_mm256_mask_reduce_min_epu16(__mmask16 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_min_epu16
// CHECK:    select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
// CHECK:    call {{.*}}i16 @llvm.vector.reduce.umin.v16i16(<16 x i16> %{{.*}})
  return _mm256_mask_reduce_min_epu16(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_min_epu16((__mmask16){0b1111111100000000}, (__m256i)(__v16hu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 9);
TEST_CONSTEXPR(_mm256_mask_reduce_min_epu16((__mmask16){0b0000000011111111}, (__m256i)(__v16hu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 1);

signed char test_mm_reduce_max_epi8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_max_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.smax.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_max_epi8(__W);
}
TEST_CONSTEXPR(_mm_reduce_max_epi8((__m128i)(__v16qs){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == 8);

signed char test_mm_reduce_min_epi8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_min_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.smin.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_min_epi8(__W);
}
TEST_CONSTEXPR(_mm_reduce_min_epi8((__m128i)(__v16qs){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == -8);

unsigned char test_mm_reduce_max_epu8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_max_epu8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.umax.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_max_epu8(__W);
}
TEST_CONSTEXPR(_mm_reduce_max_epu8((__m128i)(__v16qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 16);

unsigned char test_mm_reduce_min_epu8(__m128i __W){
// CHECK-LABEL: test_mm_reduce_min_epu8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.umin.v16i8(<16 x i8> %{{.*}})
  return _mm_reduce_min_epu8(__W);
}
TEST_CONSTEXPR(_mm_reduce_min_epu8((__m128i)(__v16qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 1);

signed char test_mm_mask_reduce_max_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_max_epi8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.smax.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_max_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_max_epi8((__mmask16)0b1111111100000000, (__m128i)(__v16qs){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == 8);
TEST_CONSTEXPR(_mm_mask_reduce_max_epi8((__mmask16)0b0000000011111111, (__m128i)(__v16qs){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == -1);

signed char test_mm_mask_reduce_min_epi8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_min_epi8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.smin.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_min_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_min_epi8((__mmask16)0b1111111100000000, (__m128i)(__v16qs){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == 1);
TEST_CONSTEXPR(_mm_mask_reduce_min_epi8((__mmask16)0b0000000011111111, (__m128i)(__v16qs){-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8}) == -8);

unsigned char test_mm_mask_reduce_max_epu8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_max_epu8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.umax.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_max_epu8(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_max_epu8((__mmask16)0b1111111100000000, (__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 16);
TEST_CONSTEXPR(_mm_mask_reduce_max_epu8((__mmask16)0b0000000011111111, (__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 8);

unsigned char test_mm_mask_reduce_min_epu8(__mmask16 __M, __m128i __W){
// CHECK-LABEL: test_mm_mask_reduce_min_epu8
// CHECK:    select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.umin.v16i8(<16 x i8> %{{.*}})
  return _mm_mask_reduce_min_epu8(__M, __W);
}
TEST_CONSTEXPR(_mm_mask_reduce_min_epu8((__mmask16)0b1111111100000000, (__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 9);
TEST_CONSTEXPR(_mm_mask_reduce_min_epu8((__mmask16)0b0000000011111111, (__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 1);

signed char test_mm256_reduce_max_epi8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_max_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.smax.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_max_epi8(__W);
}
TEST_CONSTEXPR(_mm256_reduce_max_epi8((__m256i)(__v32qs){-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 16);

signed char test_mm256_reduce_min_epi8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_min_epi8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.smin.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_min_epi8(__W);
}
TEST_CONSTEXPR(_mm256_reduce_min_epi8((__m256i)(__v32qs){-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == -16);

unsigned char test_mm256_reduce_max_epu8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_max_epu8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.umax.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_max_epu8(__W);
}
TEST_CONSTEXPR(_mm256_reduce_max_epu8((__m256i)(__v32qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}) == 32);

unsigned char test_mm256_reduce_min_epu8(__m256i __W){
// CHECK-LABEL: test_mm256_reduce_min_epu8
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.umin.v32i8(<32 x i8> %{{.*}})
  return _mm256_reduce_min_epu8(__W);
}
TEST_CONSTEXPR(_mm256_reduce_min_epu8((__m256i)(__v32qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}) == 1);

signed char test_mm256_mask_reduce_max_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_max_epi8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.smax.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_max_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_max_epi8((__mmask32)0b11111111111111110000000000000000, (__m256i)(__v32qs){-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 16);
TEST_CONSTEXPR(_mm256_mask_reduce_max_epi8((__mmask32)0b00000000000000001111111111111111, (__m256i)(__v32qs){-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == -1);

signed char test_mm256_mask_reduce_min_epi8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_min_epi8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.smin.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_min_epi8(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_min_epi8((__mmask32)0b11111111111111110000000000000000, (__m256i)(__v32qs){-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == 1);
TEST_CONSTEXPR(_mm256_mask_reduce_min_epi8((__mmask32)0b00000000000000001111111111111111, (__m256i)(__v32qs){-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}) == -16);

unsigned char test_mm256_mask_reduce_max_epu8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_max_epu8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.umax.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_max_epu8(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_max_epu8((__mmask32)0b11111111111111110000000000000000, (__m256i)(__v32qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}) == 32);
TEST_CONSTEXPR(_mm256_mask_reduce_max_epu8((__mmask32)0b00000000000000001111111111111111, (__m256i)(__v32qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}) == 16);

unsigned char test_mm256_mask_reduce_min_epu8(__mmask32 __M, __m256i __W){
// CHECK-LABEL: test_mm256_mask_reduce_min_epu8
// CHECK:    select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
// CHECK:    call {{.*}}i8 @llvm.vector.reduce.umin.v32i8(<32 x i8> %{{.*}})
  return _mm256_mask_reduce_min_epu8(__M, __W);
}
TEST_CONSTEXPR(_mm256_mask_reduce_min_epu8((__mmask32)0b11111111111111110000000000000000, (__m256i)(__v32qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}) == 17);
TEST_CONSTEXPR(_mm256_mask_reduce_min_epu8((__mmask32)0b00000000000000001111111111111111, (__m256i)(__v32qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}) == 1);
