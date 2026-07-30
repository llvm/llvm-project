/*===------------ avx10_2_v2auxintrin.h - AVX10_2_V2AUX -------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error                                                                         \
    "Never use <avx10_2_v2auxintrin.h> directly; include <immintrin.h> instead."
#endif // __IMMINTRIN_H

#ifdef __SSE2__

#ifndef __AVX10_2_V2AUXINTRIN_H
#define __AVX10_2_V2AUXINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS128                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx10v2aux"),     \
                 __min_vector_width__(128)))
#define __DEFAULT_FN_ATTRS256                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx10v2aux"),     \
                 __min_vector_width__(256)))

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8 instruction.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8]. The lower 4 bytes contain the converted
///    values; the upper bytes are zeroed.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtps_bf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_128((__v4sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_mask_cvtps_bf8(__m128i __W,
                                                                   __mmask8 __U,
                                                                   __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvtps_bf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtps_bf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm_cvtps_bf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8 instruction.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8]. The lower 8 bytes contain the converted
///    values; the upper bytes are zeroed.
static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_cvtps_bf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_256((__v8sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtps_bf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvtps_bf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtps_bf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm256_cvtps_bf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8S instruction.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvts_ps_bf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_128((__v4sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_ps_bf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvts_ps_bf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_ps_bf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm_cvts_ps_bf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8S instruction.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_cvts_ps_bf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_256((__v8sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvts_ps_bf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvts_ps_bf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_ps_bf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm256_cvts_ps_bf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8 instruction.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtps_hf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_128((__v4sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_mask_cvtps_hf8(__m128i __W,
                                                                   __mmask8 __U,
                                                                   __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvtps_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtps_hf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm_cvtps_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8 instruction.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_cvtps_hf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_256((__v8sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtps_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvtps_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtps_hf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm256_cvtps_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8S instruction.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvts_ps_hf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_128((__v4sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_ps_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvts_ps_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_ps_hf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm_cvts_ps_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8S instruction.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_cvts_ps_hf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_256((__v8sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvts_ps_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvts_ps_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements with saturation, and store
///    the results in a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2HF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_ps_hf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm256_cvts_ps_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd, and
///    store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8 instruction.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtrops_hf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_128((__v4sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd, and
///    store the results in a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtrops_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvtrops_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd, and
///    store the results in a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtrops_hf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm_cvtrops_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd, and
///    store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8 instruction.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_cvtrops_hf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_256((__v8sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd, and
///    store the results in a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtrops_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvtrops_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd, and
///    store the results in a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtrops_hf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm256_cvtrops_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd with
///    saturation, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8S instruction.
///
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvts_rops_hf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_128((__v4sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd with
///    saturation, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_rops_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvts_rops_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd with
///    saturation, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_rops_hf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm_cvts_rops_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd with
///    saturation, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8S instruction.
///
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvts_rops_hf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_256((__v8sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd with
///    saturation, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvts_rops_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvts_rops_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd with
///    saturation, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_rops_hf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm256_cvts_rops_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8 instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtbiasps_bf8(__m128i __A,
                                                                  __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_128((__v16qi)__A, (__v4sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtbiasps_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvtbiasps_bf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbiasps_bf8(__mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvtbiasps_bf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8 instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtbiasps_bf8(__m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_256((__v32qi)__A, (__v8sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbiasps_bf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvtbiasps_bf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbiasps_bf8(__mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvtbiasps_bf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8S instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvts_biasps_bf8(__m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_128((__v16qi)__A, (__v4sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_biasps_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvts_biasps_bf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_biasps_bf8(__mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvts_biasps_bf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8S instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvts_biasps_bf8(__m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_256((__v32qi)__A, (__v8sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_mask_cvts_biasps_bf8(
    __m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvts_biasps_bf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_biasps_bf8(__mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvts_biasps_bf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8 instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtbiasps_hf8(__m128i __A,
                                                                  __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_128((__v16qi)__A, (__v4sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtbiasps_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvtbiasps_hf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbiasps_hf8(__mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvtbiasps_hf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8 instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtbiasps_hf8(__m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_256((__v32qi)__A, (__v8sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbiasps_hf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvtbiasps_hf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8 instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbiasps_hf8(__mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvtbiasps_hf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8S instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvts_biasps_hf8(__m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_128((__v16qi)__A, (__v4sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_biasps_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvts_biasps_hf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing bias values.
/// \param __B
///    A 128-bit vector of [4 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_biasps_hf8(__mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm_cvts_biasps_hf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8S instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvts_biasps_hf8(__m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_256((__v32qi)__A, (__v8sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_mask_cvts_biasps_hf8(
    __m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvts_biasps_hf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements with saturation using bias
///    values from \a __A, and store the results using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8S instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [32 x i8] containing bias values.
/// \param __B
///    A 256-bit vector of [8 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_biasps_hf8(__mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm256_cvts_biasps_hf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 128-bit vector of [4 x float] containing the converted values.
static __inline__ __m128 __DEFAULT_FN_ATTRS128 _mm_cvtbf8_ps(__m128i __A) {
  return (__m128)__builtin_ia32_vcvtbf82ps_128((__v16qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __W
///    A 128-bit vector of [4 x float] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 128-bit vector of [4 x float] containing the converted values.
static __inline__ __m128 __DEFAULT_FN_ATTRS128 _mm_mask_cvtbf8_ps(__m128 __W,
                                                                  __mmask8 __U,
                                                                  __m128i __A) {
  return (__m128)__builtin_ia32_selectps_128(
      (__mmask8)__U, (__v4sf)_mm_cvtbf8_ps(__A), (__v4sf)__W);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 128-bit vector of [4 x float] containing the converted values.
static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbf8_ps(__mmask8 __U, __m128i __A) {
  return (__m128)__builtin_ia32_selectps_128(
      (__mmask8)__U, (__v4sf)_mm_cvtbf8_ps(__A), (__v4sf)_mm_setzero_ps());
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 256-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 256-bit vector of [8 x float] containing the converted values.
static __inline__ __m256 __DEFAULT_FN_ATTRS256 _mm256_cvtbf8_ps(__m128i __A) {
  return (__m256)__builtin_ia32_vcvtbf82ps_256((__v16qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __W
///    A 256-bit vector of [8 x float] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 256-bit vector of [8 x float] containing the converted values.
static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbf8_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  return (__m256)__builtin_ia32_selectps_256(
      (__mmask8)__U, (__v8sf)_mm256_cvtbf8_ps(__A), (__v8sf)__W);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 256-bit vector of [8 x float] containing the converted values.
static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbf8_ps(__mmask8 __U, __m128i __A) {
  return (__m256)__builtin_ia32_selectps_256((__mmask8)__U,
                                             (__v8sf)_mm256_cvtbf8_ps(__A),
                                             (__v8sf)_mm256_setzero_ps());
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 128-bit vector of [4 x float] containing the converted values.
static __inline__ __m128 __DEFAULT_FN_ATTRS128 _mm_cvthf8_ps(__m128i __A) {
  return (__m128)__builtin_ia32_vcvthf82ps_128((__v16qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __W
///    A 128-bit vector of [4 x float] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 128-bit vector of [4 x float] containing the converted values.
static __inline__ __m128 __DEFAULT_FN_ATTRS128 _mm_mask_cvthf8_ps(__m128 __W,
                                                                  __mmask8 __U,
                                                                  __m128i __A) {
  return (__m128)__builtin_ia32_selectps_128(
      (__mmask8)__U, (__v4sf)_mm_cvthf8_ps(__A), (__v4sf)__W);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 128-bit vector of [4 x float] containing the converted values.
static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_maskz_cvthf8_ps(__mmask8 __U, __m128i __A) {
  return (__m128)__builtin_ia32_selectps_128(
      (__mmask8)__U, (__v4sf)_mm_cvthf8_ps(__A), (__v4sf)_mm_setzero_ps());
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 256-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 256-bit vector of [8 x float] containing the converted values.
static __inline__ __m256 __DEFAULT_FN_ATTRS256 _mm256_cvthf8_ps(__m128i __A) {
  return (__m256)__builtin_ia32_vcvthf82ps_256((__v16qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __W
///    A 256-bit vector of [8 x float] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 256-bit vector of [8 x float] containing the converted values.
static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_mask_cvthf8_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  return (__m256)__builtin_ia32_selectps_256(
      (__mmask8)__U, (__v8sf)_mm256_cvthf8_ps(__A), (__v8sf)__W);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 256-bit vector of [8 x float] containing the converted values.
static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_maskz_cvthf8_ps(__mmask8 __U, __m128i __A) {
  return (__m256)__builtin_ia32_selectps_256((__mmask8)__U,
                                             (__v8sf)_mm256_cvthf8_ps(__A),
                                             (__v8sf)_mm256_setzero_ps());
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    BF6 (6-bit) floating-point elements with saturation, and store the
///    results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82BF6S instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted BF6 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtbf8_bf6s(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvtbf82bf6s_128((__v16qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    BF6 (6-bit) floating-point elements with saturation, and store the
///    results in a 256-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82BF6S instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF8 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted BF6 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtbf8_bf6s(__m256i __A) {
  return (__m256i)__builtin_ia32_vcvtbf82bf6s_256((__v32qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    HF6 (6-bit) floating-point elements with saturation, and store the
///    results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82HF6S instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF6 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvthf8_hf6s(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvthf82hf6s_128((__v16qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    HF6 (6-bit) floating-point elements with saturation, and store the
///    results in a 256-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82HF6S instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing HF8 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF6 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvthf8_hf6s(__m256i __A) {
  return (__m256i)__builtin_ia32_vcvthf82hf6s_256((__v32qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82BF4S instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted BF4 values
///    (lower 8 bytes used).
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtbf8_bf4s(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvtbf82bf4s_128((__v16qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82BF4S instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF8 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted BF4 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtbf8_bf4s(__m256i __A) {
  return (__m128i)__builtin_ia32_vcvtbf82bf4s_256((__v32qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82BF4S instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted BF4 values
///    (lower 8 bytes used).
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvthf8_bf4s(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvthf82bf4s_128((__v16qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82BF4S instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing HF8 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted BF4 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvthf8_bf4s(__m256i __A) {
  return (__m128i)__builtin_ia32_vcvthf82bf4s_256((__v32qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results to memory at \a __P.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82BF4S instruction.
///
/// \param __P
///    A pointer to a 64-bit memory location. The address does not need to be
///    aligned.
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
static __inline__ void __DEFAULT_FN_ATTRS128
_mm_cvtbf8_bf4s_storeu(void *__P, __m128i __A) {
  __builtin_ia32_vcvtbf82bf4s_128_mem(__P, (__v16qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results to memory at \a __P.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82BF4S instruction.
///
/// \param __P
///    A pointer to a 128-bit memory location. The address does not need to be
///    aligned.
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF8 values.
static __inline__ void __DEFAULT_FN_ATTRS256
_mm256_cvtbf8_bf4s_storeu(void *__P, __m256i __A) {
  __builtin_ia32_vcvtbf82bf4s_256_mem(__P, (__v32qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results to memory at \a __P.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82BF4S instruction.
///
/// \param __P
///    A pointer to a 64-bit memory location. The address does not need to be
///    aligned.
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
static __inline__ void __DEFAULT_FN_ATTRS128
_mm_cvthf8_bf4s_storeu(void *__P, __m128i __A) {
  __builtin_ia32_vcvthf82bf4s_128_mem(__P, (__v16qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results to memory at \a __P.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82BF4S instruction.
///
/// \param __P
///    A pointer to a 128-bit memory location. The address does not need to be
///    aligned.
/// \param __A
///    A 256-bit vector of [32 x i8] containing HF8 values.
static __inline__ void __DEFAULT_FN_ATTRS256
_mm256_cvthf8_bf4s_storeu(void *__P, __m256i __A) {
  __builtin_ia32_vcvthf82bf4s_256_mem(__P, (__v32qi)__A);
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 128-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF4 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtbf4_hf8(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvtbf42hf8_128((__v16qi)__A);
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF4 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtbf4_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(__U, (__v16qi)_mm_cvtbf4_hf8(__A),
                                             (__v16qi)__W);
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF4 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbf4_hf8(__mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(__U, (__v16qi)_mm_cvtbf4_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 256-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF4 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256 _mm256_cvtbf4_hf8(__m128i __A) {
  return (__m256i)__builtin_ia32_vcvtbf42hf8_256((__v16qi)__A);
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __W
///    A 256-bit vector of [32 x i8] used for writemask.
/// \param __U
///    A 32-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF4 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbf4_hf8(__m256i __W, __mmask32 __U, __m128i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvtbf4_hf8(__A), (__v32qi)__W);
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __U
///    A 32-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF4 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbf4_hf8(__mmask32 __U, __m128i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvtbf4_hf8(__A), (__v32qi)_mm256_setzero_si256());
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 128-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF6 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtbf6_hf8(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvtbf62hf8_128((__v16qi)__A);
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF6 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtbf6_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(__U, (__v16qi)_mm_cvtbf6_hf8(__A),
                                             (__v16qi)__W);
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF6 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbf6_hf8(__mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(__U, (__v16qi)_mm_cvtbf6_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 256-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF6 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256 _mm256_cvtbf6_hf8(__m256i __A) {
  return (__m256i)__builtin_ia32_vcvtbf62hf8_256((__v32qi)__A);
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __W
///    A 256-bit vector of [32 x i8] used for writemask.
/// \param __U
///    A 32-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF6 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbf6_hf8(__m256i __W, __mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvtbf6_hf8(__A), (__v32qi)__W);
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __U
///    A 32-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF6 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbf6_hf8(__mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvtbf6_hf8(__A), (__v32qi)_mm256_setzero_si256());
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 128-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF6 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvthf6_hf8(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvthf62hf8_128((__v16qi)__A);
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF6 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvthf6_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(__U, (__v16qi)_mm_cvthf6_hf8(__A),
                                             (__v16qi)__W);
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF6 values.
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted HF8 values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvthf6_hf8(__mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(__U, (__v16qi)_mm_cvthf6_hf8(__A),
                                             (__v16qi)_mm_setzero_si128());
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 256-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing HF6 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256 _mm256_cvthf6_hf8(__m256i __A) {
  return (__m256i)__builtin_ia32_vcvthf62hf8_256((__v32qi)__A);
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __W
///    A 256-bit vector of [32 x i8] used for writemask.
/// \param __U
///    A 32-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [32 x i8] containing HF6 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_cvthf6_hf8(__m256i __W, __mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvthf6_hf8(__A), (__v32qi)__W);
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results using
///    zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __U
///    A 32-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [32 x i8] containing HF6 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted HF8 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvthf6_hf8(__mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvthf6_hf8(__A), (__v32qi)_mm256_setzero_si256());
}

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param A
///    A 128-bit vector of [16 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 128-bit vector of [16 x i8] containing the unpacked values.
#define _mm_unpackb_epi8(A, imm)                                               \
  ((__m128i)__builtin_ia32_vunpackb128((__v16qi)(__m128i)(A), (int)(imm)))

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 128-bit vector using writemask \a U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param U
///    A 16-bit mask indicating which elements to write.
/// \param A
///    A 128-bit vector of [16 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 128-bit vector of [16 x i8] containing the unpacked values.
#define _mm_mask_unpackb_epi8(W, U, A, imm)                                    \
  ((__m128i)__builtin_ia32_selectb_128((__mmask16)(U),                         \
                                       (__v16qi)_mm_unpackb_epi8((A), (imm)),  \
                                       (__v16qi)(__m128i)(W)))

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 128-bit vector using zeromask \a U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param A
///    A 128-bit vector of [16 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 128-bit vector of [16 x i8] containing the unpacked values.
#define _mm_maskz_unpackb_epi8(U, A, imm)                                      \
  ((__m128i)__builtin_ia32_selectb_128((__mmask16)(U),                         \
                                       (__v16qi)_mm_unpackb_epi8((A), (imm)),  \
                                       (__v16qi)_mm_setzero_si128()))

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 256-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param A
///    A 256-bit vector of [32 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 256-bit vector of [32 x i8] containing the unpacked values.
#define _mm256_unpackb_epi8(A, imm)                                            \
  ((__m256i)__builtin_ia32_vunpackb256((__v32qi)(__m256i)(A), (int)(imm)))

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 256-bit vector using writemask \a U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param W
///    A 256-bit vector of [32 x i8] used for writemask.
/// \param U
///    A 32-bit mask indicating which elements to write.
/// \param A
///    A 256-bit vector of [32 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 256-bit vector of [32 x i8] containing the unpacked values.
#define _mm256_mask_unpackb_epi8(W, U, A, imm)                                 \
  ((__m256i)__builtin_ia32_selectb_256(                                        \
      (__mmask32)(U), (__v32qi)_mm256_unpackb_epi8((A), (imm)),                \
      (__v32qi)(__m256i)(W)))

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 256-bit vector using zeromask \a U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param U
///    A 32-bit mask indicating which elements to write (zero otherwise).
/// \param A
///    A 256-bit vector of [32 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 256-bit vector of [32 x i8] containing the unpacked values.
#define _mm256_maskz_unpackb_epi8(U, A, imm)                                   \
  ((__m256i)__builtin_ia32_selectb_256(                                        \
      (__mmask32)(U), (__v32qi)_mm256_unpackb_epi8((A), (imm)),                \
      (__v32qi)_mm256_setzero_si256()))

/* VPMOVSSDB - Symmetric Signed Saturation DWord to Byte */

/// Convert packed signed 32-bit integers in \a __A to packed 8-bit integers
///    with symmetric signed saturation (clamp to [-127, +127]), and store
///    the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMOVSSDB instruction.
///
/// \param __A
///    A 128-bit vector of [4 x i32].
/// \returns
///    A 128-bit vector of [16 x i8]. The lower 4 bytes contain the converted
///    values; the upper bytes are zeroed.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtss_epi32_epi8(__m128i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb128_mask(
      (__v4si)__A, (__v16qi)_mm_setzero_si128(), (__mmask8)-1);
}

/// Convert packed signed 32-bit integers in \a __A to packed 8-bit integers
///    with symmetric signed saturation, using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMOVSSDB instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [4 x i32].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtss_epi32_epi8(__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb128_mask((__v4si)__A, (__v16qi)__W,
                                                   __U);
}

/// Convert packed signed 32-bit integers in \a __A to packed 8-bit integers
///    with symmetric signed saturation, using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMOVSSDB instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [4 x i32].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtss_epi32_epi8(__mmask8 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb128_mask(
      (__v4si)__A, (__v16qi)_mm_setzero_si128(), __U);
}

/// Convert packed signed 32-bit integers in \a __A to packed 8-bit integers
///    with symmetric signed saturation (clamp to [-127, +127]), and store
///    the results in a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMOVSSDB instruction.
///
/// \param __A
///    A 256-bit vector of [8 x i32].
/// \returns
///    A 128-bit vector of [16 x i8]. The lower 8 bytes contain the converted
///    values; the upper bytes are zeroed.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtss_epi32_epi8(__m256i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb256_mask(
      (__v8si)__A, (__v16qi)_mm_setzero_si128(), (__mmask8)-1);
}

/// Convert packed signed 32-bit integers in \a __A to packed 8-bit integers
///    with symmetric signed saturation, using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMOVSSDB instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [8 x i32].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtss_epi32_epi8(__m128i __W, __mmask8 __U, __m256i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb256_mask((__v8si)__A, (__v16qi)__W,
                                                   __U);
}

/// Convert packed signed 32-bit integers in \a __A to packed 8-bit integers
///    with symmetric signed saturation, using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMOVSSDB instruction.
///
/// \param __U
///    A 8-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [8 x i32].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtss_epi32_epi8(__mmask8 __U, __m256i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb256_mask(
      (__v8si)__A, (__v16qi)_mm_setzero_si128(), __U);
}

/// Truncate packed 32-bit integers in \a __A to packed 8-bit integers with
/// symmetric signed saturation, and store the results to memory at \a __P
/// using writemask \a __M (elements not selected by the mask are not written).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMOVSSDB instruction.
///
/// \param __P
///    Pointer to the destination memory.
/// \param __M
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [4 x i32].
static __inline__ void __DEFAULT_FN_ATTRS128
_mm_mask_cvtss_epi32_storeu_epi8(void *__P, __mmask8 __M, __m128i __A) {
  __builtin_ia32_vpmovssdb128mem_mask((__v16qi *)__P, (__v4si)__A, __M);
}

/// Truncate packed 32-bit integers in \a __A to packed 8-bit integers with
/// symmetric signed saturation, and store the results to memory at \a __P
/// using writemask \a __M (elements not selected by the mask are not written).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VPMOVSSDB instruction.
///
/// \param __P
///    Pointer to the destination memory.
/// \param __M
///    A 8-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [8 x i32].
static __inline__ void __DEFAULT_FN_ATTRS256
_mm256_mask_cvtss_epi32_storeu_epi8(void *__P, __mmask8 __M, __m256i __A) {
  __builtin_ia32_vpmovssdb256mem_mask((__v16qi *)__P, (__v8si)__A, __M);
}

#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256

#endif // __AVX10_2_V2AUXINTRIN_H
#endif // __SSE2__
