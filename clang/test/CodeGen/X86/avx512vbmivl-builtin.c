// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vbmi -target-feature +avx512vl -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

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

