// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>


__m128d test_mm_mask_expand_pd(__m128d __W, __mmask8 __U, __m128d __A) {

  return _mm_mask_expand_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_expand_pd(__mmask8 __U, __m128d __A) {

  return _mm_maskz_expand_pd(__U,__A); 
}
__m256d test_mm256_mask_expand_pd(__m256d __W, __mmask8 __U, __m256d __A) {

  return _mm256_mask_expand_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_expand_pd(__mmask8 __U, __m256d __A) {

  return _mm256_maskz_expand_pd(__U,__A); 
}
__m128i test_mm_mask_expand_epi64(__m128i __W, __mmask8 __U, __m128i __A) {

  return _mm_mask_expand_epi64(__W,__U,__A); 
}
__m128i test_mm_maskz_expand_epi64(__mmask8 __U, __m128i __A) {

  return _mm_maskz_expand_epi64(__U,__A); 
}
__m256i test_mm256_mask_expand_epi64(__m256i __W, __mmask8 __U, __m256i __A) {

  return _mm256_mask_expand_epi64(__W,__U,__A); 
}
__m256i test_mm256_maskz_expand_epi64(__mmask8 __U, __m256i __A) {

  return _mm256_maskz_expand_epi64(__U,__A); 
}

__m128 test_mm_mask_expand_ps(__m128 __W, __mmask8 __U, __m128 __A) {

  return _mm_mask_expand_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_expand_ps(__mmask8 __U, __m128 __A) {

  return _mm_maskz_expand_ps(__U,__A); 
}
__m256 test_mm256_mask_expand_ps(__m256 __W, __mmask8 __U, __m256 __A) {

  return _mm256_mask_expand_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_expand_ps(__mmask8 __U, __m256 __A) {

  return _mm256_maskz_expand_ps(__U,__A); 
}
__m128i test_mm_mask_expand_epi32(__m128i __W, __mmask8 __U, __m128i __A) {

  return _mm_mask_expand_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_expand_epi32(__mmask8 __U, __m128i __A) {

  return _mm_maskz_expand_epi32(__U,__A); 
}
__m256i test_mm256_mask_expand_epi32(__m256i __W, __mmask8 __U, __m256i __A) {

  return _mm256_mask_expand_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_expand_epi32(__mmask8 __U, __m256i __A) {

  return _mm256_maskz_expand_epi32(__U,__A); 
}

__m128d test_mm_mask_compress_pd(__m128d __W, __mmask8 __U, __m128d __A) {

  return _mm_mask_compress_pd(__W,__U,__A); 
}

__m128d test_mm_maskz_compress_pd(__mmask8 __U, __m128d __A) {

  return _mm_maskz_compress_pd(__U,__A); 
}

__m256d test_mm256_mask_compress_pd(__m256d __W, __mmask8 __U, __m256d __A) {

  return _mm256_mask_compress_pd(__W,__U,__A); 
}

__m256d test_mm256_maskz_compress_pd(__mmask8 __U, __m256d __A) {

  return _mm256_maskz_compress_pd(__U,__A); 
}

__m128i test_mm_mask_compress_epi64(__m128i __W, __mmask8 __U, __m128i __A) {

  return _mm_mask_compress_epi64(__W,__U,__A); 
}

__m128i test_mm_maskz_compress_epi64(__mmask8 __U, __m128i __A) {

  return _mm_maskz_compress_epi64(__U,__A); 
}

__m256i test_mm256_mask_compress_epi64(__m256i __W, __mmask8 __U, __m256i __A) {

  return _mm256_mask_compress_epi64(__W,__U,__A); 
}

__m256i test_mm256_maskz_compress_epi64(__mmask8 __U, __m256i __A) {

  return _mm256_maskz_compress_epi64(__U,__A); 
}

__m128 test_mm_mask_compress_ps(__m128 __W, __mmask8 __U, __m128 __A) {

  return _mm_mask_compress_ps(__W,__U,__A); 
}

__m128 test_mm_maskz_compress_ps(__mmask8 __U, __m128 __A) {

  return _mm_maskz_compress_ps(__U,__A); 
}

__m256 test_mm256_mask_compress_ps(__m256 __W, __mmask8 __U, __m256 __A) {

  return _mm256_mask_compress_ps(__W,__U,__A); 
}

__m256 test_mm256_maskz_compress_ps(__mmask8 __U, __m256 __A) {

  return _mm256_maskz_compress_ps(__U,__A); 
}

__m128i test_mm_mask_compress_epi32(__m128i __W, __mmask8 __U, __m128i __A) {

  return _mm_mask_compress_epi32(__W,__U,__A); 
}

__m128i test_mm_maskz_compress_epi32(__mmask8 __U, __m128i __A) {

  return _mm_maskz_compress_epi32(__U,__A); 
}

__m256i test_mm256_mask_compress_epi32(__m256i __W, __mmask8 __U, __m256i __A) {

  return _mm256_mask_compress_epi32(__W,__U,__A); 
}

__m256i test_mm256_maskz_compress_epi32(__mmask8 __U, __m256i __A) {

  return _mm256_maskz_compress_epi32(__U,__A); 
}
