// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s


#include <immintrin.h>
#include "builtin_test_helpers.h"

__m512i test_mm512_mask2_permutex2var_epi8(__m512i __A, __m512i __I, __mmask64 __U, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask2_permutex2var_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.vpermi2var.qi.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask2_permutex2var_epi8(__A, __I, __U, __B); 
}

__m512i test_mm512_permutex2var_epi8(__m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: test_mm512_permutex2var_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.vpermi2var.qi.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_permutex2var_epi8(__A, __I, __B); 
}

__m512i test_mm512_mask_permutex2var_epi8(__m512i __A, __mmask64 __U, __m512i __I, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_permutex2var_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.vpermi2var.qi.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_permutex2var_epi8(__A, __U, __I, __B); 
}

__m512i test_mm512_maskz_permutex2var_epi8(__mmask64 __U, __m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_permutex2var_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.vpermi2var.qi.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_permutex2var_epi8(__U, __A, __I, __B); 
}

TEST_CONSTEXPR(match_v64qu(
    _mm512_permutex2var_epi8((__m512i)(__v64qu){
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63},
        (__m512i)(__v64qu){
            0, 64, 1, 65, 2, 66, 3, 67,
            4, 68, 5, 69, 6, 70, 7, 71,
            8, 72, 9, 73, 10, 74, 11, 75,
            12, 76, 13, 77, 14, 78, 15, 79,
            16, 80, 17, 81, 18, 82, 19, 83,
            20, 84, 21, 85, 22, 86, 23, 87,
            24, 88, 25, 89, 26, 90, 27, 91,
            28, 92, 29, 93, 30, 94, 31, 95},
        (__m512i)(__v64qu){
            200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215,
            216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 227, 228, 229, 230, 231,
            232, 233, 234, 235, 236, 237, 238, 239,
            240, 241, 242, 243, 244, 245, 246, 247,
            248, 249, 250, 251, 252, 253, 254, 255,
            0, 1, 2, 3, 4, 5, 6, 7}),
    0, 200, 1, 201, 2, 202, 3, 203,
    4, 204, 5, 205, 6, 206, 7, 207,
    8, 208, 9, 209, 10, 210, 11, 211,
    12, 212, 13, 213, 14, 214, 15, 215,
    16, 216, 17, 217, 18, 218, 19, 219,
    20, 220, 21, 221, 22, 222, 23, 223,
    24, 224, 25, 225, 26, 226, 27, 227,
    28, 228, 29, 229, 30, 230, 31, 231));
TEST_CONSTEXPR(match_v64qu(
    _mm512_mask_permutex2var_epi8((__m512i)(__v64qu){
            10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65,
            66, 67, 68, 69, 70, 71, 72, 73},
        0xAAAAAAAAAAAAAAAAULL,
        (__m512i)(__v64qu){
            0, 64, 1, 65, 2, 66, 3, 67,
            4, 68, 5, 69, 6, 70, 7, 71,
            8, 72, 9, 73, 10, 74, 11, 75,
            12, 76, 13, 77, 14, 78, 15, 79,
            16, 80, 17, 81, 18, 82, 19, 83,
            20, 84, 21, 85, 22, 86, 23, 87,
            24, 88, 25, 89, 26, 90, 27, 91,
            28, 92, 29, 93, 30, 94, 31, 95},
        (__m512i)(__v64qu){
            200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215,
            216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 227, 228, 229, 230, 231,
            232, 233, 234, 235, 236, 237, 238, 239,
            240, 241, 242, 243, 244, 245, 246, 247,
            248, 249, 250, 251, 252, 253, 254, 255,
            0, 1, 2, 3, 4, 5, 6, 7}),
    10, 200, 12, 201, 14, 202, 16, 203,
    18, 204, 20, 205, 22, 206, 24, 207,
    26, 208, 28, 209, 30, 210, 32, 211,
    34, 212, 36, 213, 38, 214, 40, 215,
    42, 216, 44, 217, 46, 218, 48, 219,
    50, 220, 52, 221, 54, 222, 56, 223,
    58, 224, 60, 225, 62, 226, 64, 227,
    66, 228, 68, 229, 70, 230, 72, 231));
TEST_CONSTEXPR(match_v64qu(
    _mm512_maskz_permutex2var_epi8(0x5555555555555555ULL,
        (__m512i)(__v64qu){
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63},
        (__m512i)(__v64qu){
            0, 64, 1, 65, 2, 66, 3, 67,
            4, 68, 5, 69, 6, 70, 7, 71,
            8, 72, 9, 73, 10, 74, 11, 75,
            12, 76, 13, 77, 14, 78, 15, 79,
            16, 80, 17, 81, 18, 82, 19, 83,
            20, 84, 21, 85, 22, 86, 23, 87,
            24, 88, 25, 89, 26, 90, 27, 91,
            28, 92, 29, 93, 30, 94, 31, 95},
        (__m512i)(__v64qu){
            200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215,
            216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 227, 228, 229, 230, 231,
            232, 233, 234, 235, 236, 237, 238, 239,
            240, 241, 242, 243, 244, 245, 246, 247,
            248, 249, 250, 251, 252, 253, 254, 255,
            0, 1, 2, 3, 4, 5, 6, 7}),
    0, 0, 1, 0, 2, 0, 3, 0,
    4, 0, 5, 0, 6, 0, 7, 0,
    8, 0, 9, 0, 10, 0, 11, 0,
    12, 0, 13, 0, 14, 0, 15, 0,
    16, 0, 17, 0, 18, 0, 19, 0,
    20, 0, 21, 0, 22, 0, 23, 0,
    24, 0, 25, 0, 26, 0, 27, 0,
    28, 0, 29, 0, 30, 0, 31, 0));
TEST_CONSTEXPR(match_v64qu(
    _mm512_mask2_permutex2var_epi8((__m512i)(__v64qu){
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63},
        (__m512i)(__v64qu){
            0, 64, 1, 65, 2, 66, 3, 67,
            4, 68, 5, 69, 6, 70, 7, 71,
            8, 72, 9, 73, 10, 74, 11, 75,
            12, 76, 13, 77, 14, 78, 15, 79,
            16, 80, 17, 81, 18, 82, 19, 83,
            20, 84, 21, 85, 22, 86, 23, 87,
            24, 88, 25, 89, 26, 90, 27, 91,
            28, 92, 29, 93, 30, 94, 31, 95},
        0x5555555555555555ULL,
        (__m512i)(__v64qu){
            200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215,
            216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 227, 228, 229, 230, 231,
            232, 233, 234, 235, 236, 237, 238, 239,
            240, 241, 242, 243, 244, 245, 246, 247,
            248, 249, 250, 251, 252, 253, 254, 255,
            0, 1, 2, 3, 4, 5, 6, 7}),
    0, 64, 1, 65, 2, 66, 3, 67,
    4, 68, 5, 69, 6, 70, 7, 71,
    8, 72, 9, 73, 10, 74, 11, 75,
    12, 76, 13, 77, 14, 78, 15, 79,
    16, 80, 17, 81, 18, 82, 19, 83,
    20, 84, 21, 85, 22, 86, 23, 87,
    24, 88, 25, 89, 26, 90, 27, 91,
    28, 92, 29, 93, 30, 94, 31, 95));

__m512i test_mm512_permutexvar_epi8(__m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_permutexvar_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.permvar.qi.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_permutexvar_epi8(__A, __B); 
}

__m512i test_mm512_maskz_permutexvar_epi8(__mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_maskz_permutexvar_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.permvar.qi.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_permutexvar_epi8(__M, __A, __B); 
}

__m512i test_mm512_mask_permutexvar_epi8(__m512i __W, __mmask64 __M, __m512i __A, __m512i __B) {
  // CHECK-LABEL: test_mm512_mask_permutexvar_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.permvar.qi.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_permutexvar_epi8(__W, __M, __A, __B); 
}

__m512i test_mm512_mask_multishift_epi64_epi8(__m512i __W, __mmask64 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_mask_multishift_epi64_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.pmultishift.qb.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_multishift_epi64_epi8(__W, __M, __X, __Y); 
}

__m512i test_mm512_maskz_multishift_epi64_epi8(__mmask64 __M, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_maskz_multishift_epi64_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.pmultishift.qb.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_multishift_epi64_epi8(__M, __X, __Y); 
}

__m512i test_mm512_multishift_epi64_epi8(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: test_mm512_multishift_epi64_epi8
  // CHECK: call <64 x i8> @llvm.x86.avx512.pmultishift.qb.512(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_multishift_epi64_epi8(__X, __Y); 
}
