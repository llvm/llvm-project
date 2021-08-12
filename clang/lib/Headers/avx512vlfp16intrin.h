/*===---------- avx512vlfp16intrin.h - AVX512-FP16 intrinsics --------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error                                                                         \
    "Never use <avx512vlfp16intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VLFP16INTRIN_H
#define __AVX512VLFP16INTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS256                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512fp16, avx512vl"),                           \
                 __min_vector_width__(256)))
#define __DEFAULT_FN_ATTRS128                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512fp16, avx512vl"),                           \
                 __min_vector_width__(128)))

static __inline__ _Float16 __DEFAULT_FN_ATTRS128 _mm_cvtsh_h(__m128h __a) {
  return __a[0];
}

static __inline__ _Float16 __DEFAULT_FN_ATTRS256 _mm256_cvtsh_h(__m256h __a) {
  return __a[0];
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_set_sh(_Float16 __h) {
  return __extension__(__m128h){__h, 0, 0, 0, 0, 0, 0, 0};
}

static __inline __m128h __DEFAULT_FN_ATTRS128 _mm_set1_ph(_Float16 __h) {
  return (__m128h)(__v8hf){__h, __h, __h, __h, __h, __h, __h, __h};
}

static __inline __m256h __DEFAULT_FN_ATTRS256 _mm256_set1_ph(_Float16 __h) {
  return (__m256h)(__v16hf){__h, __h, __h, __h, __h, __h, __h, __h,
                            __h, __h, __h, __h, __h, __h, __h, __h};
}

static __inline __m128h __DEFAULT_FN_ATTRS128
_mm_set_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
           _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8) {
  return (__m128h)(__v8hf){__h8, __h7, __h6, __h5, __h4, __h3, __h2, __h1};
}

static __inline __m256h __DEFAULT_FN_ATTRS256
_mm256_set_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
              _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8,
              _Float16 __h9, _Float16 __h10, _Float16 __h11, _Float16 __h12,
              _Float16 __h13, _Float16 __h14, _Float16 __h15, _Float16 __h16) {
  return (__m256h)(__v16hf){__h16, __h15, __h14, __h13, __h12, __h11,
                            __h10, __h9,  __h8,  __h7,  __h6,  __h5,
                            __h4,  __h3,  __h2,  __h1};
}

#define _mm_setr_ph(h1, h2, h3, h4, h5, h6, h7, h8)                            \
  _mm_set_ph((h8), (h7), (h6), (h5), (h4), (h3), (h2), (h1))

#define _mm256_setr_ph(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, \
                       h14, h15, h16)                                          \
  _mm256_set_ph((h16), (h15), (h14), (h13), (h12), (h11), (h10), (h9), (h8),   \
                (h7), (h6), (h5), (h4), (h3), (h2), (h1))

static __inline__ __m256h __DEFAULT_FN_ATTRS256 _mm256_abs_ph(__m256h __A) {
  return (__m256h)_mm256_and_epi32(_mm256_set1_epi32(0x7FFF7FFF), (__m256i)__A);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_abs_ph(__m128h __A) {
  return (__m128h)_mm_and_epi32(_mm_set1_epi32(0x7FFF7FFF), (__m128i)__A);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mask_blend_ph(__mmask8 __U,
                                                                  __m128h __A,
                                                                  __m128h __W) {
  return (__m128h)__builtin_ia32_selectph_128((__mmask8)__U, (__v8hf)__W,
                                              (__v8hf)__A);
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256
_mm256_mask_blend_ph(__mmask16 __U, __m256h __A, __m256h __W) {
  return (__m256h)__builtin_ia32_selectph_256((__mmask16)__U, (__v16hf)__W,
                                              (__v16hf)__A);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128
_mm_permutex2var_ph(__m128h __A, __m128i __I, __m128h __B) {
  return (__m128h)__builtin_ia32_vpermi2varhi128((__v8hi)__A, (__v8hi)__I,
                                                 (__v8hi)__B);
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256
_mm256_permutex2var_ph(__m256h __A, __m256i __I, __m256h __B) {
  return (__m256h)__builtin_ia32_vpermi2varhi256((__v16hi)__A, (__v16hi)__I,
                                                 (__v16hi)__B);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128
_mm_permutexvar_ph(__m128i __A, __m128h __B) {
  return (__m128h)__builtin_ia32_permvarhi128((__v8hi)__B, (__v8hi)__A);
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256
_mm256_permutexvar_ph(__m256i __A, __m256h __B) {
  return (__m256h)__builtin_ia32_permvarhi256((__v16hi)__B, (__v16hi)__A);
}

#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256

#endif
