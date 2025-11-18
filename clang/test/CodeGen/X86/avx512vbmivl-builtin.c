// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s


#include <immintrin.h>
#include "builtin_test_helpers.h"

__m128i test_mm_permutexvar_epi8(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_permutexvar_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.permvar.qi.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_permutexvar_epi8(__A, __B); 
}

__m128i test_mm_maskz_permutexvar_epi8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_permutexvar_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.permvar.qi.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_permutexvar_epi8(__M, __A, __B); 
}

__m128i test_mm_mask_permutexvar_epi8(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_permutexvar_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.permvar.qi.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_permutexvar_epi8(__W, __M, __A, __B); 
}

__m256i test_mm256_permutexvar_epi8(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_permutexvar_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.permvar.qi.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_permutexvar_epi8(__A, __B); 
}

__m256i test_mm256_maskz_permutexvar_epi8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_permutexvar_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.permvar.qi.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_permutexvar_epi8(__M, __A, __B); 
}

__m256i test_mm256_mask_permutexvar_epi8(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_permutexvar_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.permvar.qi.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_permutexvar_epi8(__W, __M, __A, __B); 
}

__m128i test_mm_mask2_permutex2var_epi8(__m128i __A, __m128i __I, __mmask16 __U, __m128i __B) {
  // CHECK-LABEL: test_mm_mask2_permutex2var_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.vpermi2var.qi.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask2_permutex2var_epi8(__A, __I, __U, __B); 
}

__m256i test_mm256_mask2_permutex2var_epi8(__m256i __A, __m256i __I, __mmask32 __U, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask2_permutex2var_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.vpermi2var.qi.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask2_permutex2var_epi8(__A, __I, __U, __B); 
}

__m128i test_mm_permutex2var_epi8(__m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: test_mm_permutex2var_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.vpermi2var.qi.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_permutex2var_epi8(__A, __I, __B); 
}

__m128i test_mm_mask_permutex2var_epi8(__m128i __A, __mmask16 __U, __m128i __I, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_permutex2var_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.vpermi2var.qi.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_permutex2var_epi8(__A, __U, __I, __B); 
}

__m128i test_mm_maskz_permutex2var_epi8(__mmask16 __U, __m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_permutex2var_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.vpermi2var.qi.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_permutex2var_epi8(__U, __A, __I, __B);
}

TEST_CONSTEXPR(match_v16qu(
    _mm_permutex2var_epi8((__m128i)(__v16qu){1, 2, 3, 4, 5, 6, 7, 8,
                                             9, 10, 11, 12, 13, 14, 15, 16},
                         (__m128i)(__v16qu){0, 16, 1, 17, 2, 18, 3, 19,
                                             4, 20, 5, 21, 6, 22, 7, 23},
                         (__m128i)(__v16qu){101, 102, 103, 104, 105, 106, 107, 108,
                                            109, 110, 111, 112, 113, 114, 115, 116}),
    1, 101, 2, 102, 3, 103, 4, 104,
    5, 105, 6, 106, 7, 107, 8, 108));
TEST_CONSTEXPR(match_v16qu(
    _mm_mask_permutex2var_epi8((__m128i)(__v16qu){200, 201, 202, 203, 204, 205, 206, 207,
                                                   208, 209, 210, 211, 212, 213, 214, 215},
                               0xAAAA,
                               (__m128i)(__v16qu){0, 16, 1, 17, 2, 18, 3, 19,
                                                   4, 20, 5, 21, 6, 22, 7, 23},
                               (__m128i)(__v16qu){101, 102, 103, 104, 105, 106, 107, 108,
                                                  109, 110, 111, 112, 113, 114, 115, 116}),
    200, 101, 202, 102, 204, 103, 206, 104,
    208, 105, 210, 106, 212, 107, 214, 108));

__m256i test_mm256_permutex2var_epi8(__m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: test_mm256_permutex2var_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.vpermi2var.qi.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_permutex2var_epi8(__A, __I, __B); 
}

__m256i test_mm256_mask_permutex2var_epi8(__m256i __A, __mmask32 __U, __m256i __I, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_permutex2var_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.vpermi2var.qi.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_permutex2var_epi8(__A, __U, __I, __B); 
}

__m256i test_mm256_maskz_permutex2var_epi8(__mmask32 __U, __m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_permutex2var_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.vpermi2var.qi.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_permutex2var_epi8(__U, __A, __I, __B);
}

TEST_CONSTEXPR(match_v32qu(
    _mm256_permutex2var_epi8((__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8,
                                                 9, 10, 11, 12, 13, 14, 15, 16,
                                                 17, 18, 19, 20, 21, 22, 23, 24,
                                                 25, 26, 27, 28, 29, 30, 31, 32},
                             (__m256i)(__v32qu){0, 32, 1, 33, 2, 34, 3, 35,
                                                 4, 36, 5, 37, 6, 38, 7, 39,
                                                 8, 40, 9, 41, 10, 42, 11, 43,
                                                 12, 44, 13, 45, 14, 46, 15, 47},
                             (__m256i)(__v32qu){101, 102, 103, 104, 105, 106, 107, 108,
                                                109, 110, 111, 112, 113, 114, 115, 116,
                                                117, 118, 119, 120, 121, 122, 123, 124,
                                                125, 126, 127, 128, 129, 130, 131, 132}),
    1, 101, 2, 102, 3, 103, 4, 104,
    5, 105, 6, 106, 7, 107, 8, 108,
    9, 109, 10, 110, 11, 111, 12, 112,
    13, 113, 14, 114, 15, 115, 16, 116));
TEST_CONSTEXPR(match_v32qu(
    _mm256_mask_permutex2var_epi8((__m256i)(__v32qu){200, 201, 202, 203, 204, 205, 206, 207,
                                                      208, 209, 210, 211, 212, 213, 214, 215,
                                                      216, 217, 218, 219, 220, 221, 222, 223,
                                                      224, 225, 226, 227, 228, 229, 230, 231},
                                  0xAAAAAAAA,
                                  (__m256i)(__v32qu){0, 32, 1, 33, 2, 34, 3, 35,
                                                      4, 36, 5, 37, 6, 38, 7, 39,
                                                      8, 40, 9, 41, 10, 42, 11, 43,
                                                      12, 44, 13, 45, 14, 46, 15, 47},
                                  (__m256i)(__v32qu){101, 102, 103, 104, 105, 106, 107, 108,
                                                     109, 110, 111, 112, 113, 114, 115, 116,
                                                     117, 118, 119, 120, 121, 122, 123, 124,
                                                     125, 126, 127, 128, 129, 130, 131, 132}),
    200, 101, 202, 102, 204, 103, 206, 104,
    208, 105, 210, 106, 212, 107, 214, 108,
    216, 109, 218, 110, 220, 111, 222, 112,
    224, 113, 226, 114, 228, 115, 230, 116));

__m128i test_mm_mask_multishift_epi64_epi8(__m128i __W, __mmask16 __M, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_mask_multishift_epi64_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.pmultishift.qb.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_multishift_epi64_epi8(__W, __M, __X, __Y); 
}

__m128i test_mm_maskz_multishift_epi64_epi8(__mmask16 __M, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_maskz_multishift_epi64_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.pmultishift.qb.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_multishift_epi64_epi8(__M, __X, __Y); 
}

__m128i test_mm_multishift_epi64_epi8(__m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_multishift_epi64_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.pmultishift.qb.128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_multishift_epi64_epi8(__X, __Y); 
}

__m256i test_mm256_mask_multishift_epi64_epi8(__m256i __W, __mmask32 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_mask_multishift_epi64_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.pmultishift.qb.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_multishift_epi64_epi8(__W, __M, __X, __Y); 
}

__m256i test_mm256_maskz_multishift_epi64_epi8(__mmask32 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_maskz_multishift_epi64_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.pmultishift.qb.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_multishift_epi64_epi8(__M, __X, __Y); 
}

__m256i test_mm256_multishift_epi64_epi8(__m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_multishift_epi64_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.pmultishift.qb.256(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_multishift_epi64_epi8(__X, __Y); 
}

