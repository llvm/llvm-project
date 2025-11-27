
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vlvbmi2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>


__m128i test_mm_mask_compress_epi16(__m128i __S, __mmask8 __U, __m128i __D) {

  return _mm_mask_compress_epi16(__S, __U, __D);
}

__m128i test_mm_maskz_compress_epi16(__mmask8 __U, __m128i __D) {

  return _mm_maskz_compress_epi16(__U, __D);
}

__m128i test_mm_mask_compress_epi8(__m128i __S, __mmask16 __U, __m128i __D) {

  return _mm_mask_compress_epi8(__S, __U, __D);
}

__m128i test_mm_maskz_compress_epi8(__mmask16 __U, __m128i __D) {

  return _mm_maskz_compress_epi8(__U, __D);
}

__m128i test_mm_mask_expand_epi16(__m128i __S, __mmask8 __U, __m128i __D) {

  return _mm_mask_expand_epi16(__S, __U, __D);
}

__m128i test_mm_maskz_expand_epi16(__mmask8 __U, __m128i __D) {

  return _mm_maskz_expand_epi16(__U, __D);
}

__m128i test_mm_mask_expand_epi8(__m128i __S, __mmask16 __U, __m128i __D) {

  return _mm_mask_expand_epi8(__S, __U, __D);
}

__m128i test_mm_maskz_expand_epi8(__mmask16 __U, __m128i __D) {

  return _mm_maskz_expand_epi8(__U, __D);
}
