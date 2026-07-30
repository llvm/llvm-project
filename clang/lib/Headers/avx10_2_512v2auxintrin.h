/*===--------- avx10_2_512v2auxintrin.h - AVX10_2_512V2AUX ---------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error                                                                         \
    "Never use <avx10_2_512v2auxintrin.h> directly; include <immintrin.h> instead."
#endif // __IMMINTRIN_H

#ifdef __SSE2__

#ifndef __AVX10_2_512V2AUXINTRIN_H
#define __AVX10_2_512V2AUXINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS512                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx10v2aux"),     \
                 __min_vector_width__(512)))

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed BF8 (8-bit) floating-point elements, and store the results in
///    a 128-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTPS2BF8 instruction.
///
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512 _mm512_cvtps_bf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_512((__v16sf)__A);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtps_bf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvtps_bf8(__A), (__v16qi)__W);
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
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtps_bf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm512_cvtps_bf8(__A),
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
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512 _mm512_cvts_ps_bf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_512((__v16sf)__A);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvts_ps_bf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvts_ps_bf8(__A), (__v16qi)__W);
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
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_ps_bf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm512_cvts_ps_bf8(__A),
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
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512 _mm512_cvtps_hf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_512((__v16sf)__A);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtps_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvtps_hf8(__A), (__v16qi)__W);
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
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtps_hf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm512_cvtps_hf8(__A),
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
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512 _mm512_cvts_ps_hf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_512((__v16sf)__A);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvts_ps_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvts_ps_hf8(__A), (__v16qi)__W);
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
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_ps_hf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm512_cvts_ps_hf8(__A),
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
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512 _mm512_cvtrops_hf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_512((__v16sf)__A);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtrops_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvtrops_hf8(__A), (__v16qi)__W);
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
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtrops_hf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm512_cvtrops_hf8(__A),
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
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvts_rops_hf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_512((__v16sf)__A);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd with
///    saturation, and store the results in a 128-bit vector using writemask
///    \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8S instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvts_rops_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvts_rops_hf8(__A), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __A
///    to packed HF8 (8-bit) floating-point elements using round-to-odd with
///    saturation, and store the results in a 128-bit vector using zeromask
///    \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTROPS2HF8S instruction.
///
/// \param __U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_rops_hf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm512_cvts_rops_hf8(__A),
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
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvtbiasps_bf8(__m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_512((__v64qi)__A, (__v16sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results in a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbiasps_bf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvtbiasps_bf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed BF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results in a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2BF8 instruction.
///
/// \param __U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbiasps_bf8(__mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvtbiasps_bf8(__A, __B),
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
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvts_biasps_bf8(__m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_512((__v64qi)__A,
                                                     (__v16sf)__B);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512 _mm512_mask_cvts_biasps_bf8(
    __m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvts_biasps_bf8(__A, __B), (__v16qi)__W);
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
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_biasps_bf8(__mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvts_biasps_bf8(__A, __B),
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
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvtbiasps_hf8(__m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_512((__v64qi)__A, (__v16sf)__B);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results in a 128-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8 instruction.
///
/// \param __W
///    A 128-bit vector of [16 x i8] used for writemask.
/// \param __U
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbiasps_hf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvtbiasps_hf8(__A, __B), (__v16qi)__W);
}

/// Convert packed single-precision (32-bit) floating-point elements in \a __B
///    to packed HF8 (8-bit) floating-point elements using bias values from
///    \a __A, and store the results in a 128-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBIASPS2HF8 instruction.
///
/// \param __U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbiasps_hf8(__mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvtbiasps_hf8(__A, __B),
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
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvts_biasps_hf8(__m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_512((__v64qi)__A,
                                                     (__v16sf)__B);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512 _mm512_mask_cvts_biasps_hf8(
    __m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvts_biasps_hf8(__A, __B), (__v16qi)__W);
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
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [64 x i8] containing bias values.
/// \param __B
///    A 512-bit vector of [16 x float].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_biasps_hf8(__mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm512_cvts_biasps_hf8(__A, __B),
      (__v16qi)_mm_setzero_si128());
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 512-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 512-bit vector of [16 x float] containing the converted values.
static __inline__ __m512 __DEFAULT_FN_ATTRS512 _mm512_cvtbf8_ps(__m128i __A) {
  return (__m512)__builtin_ia32_vcvtbf82ps_512((__v16qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 512-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __W
///    A 512-bit vector of [16 x float] used for writemask.
/// \param __U
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 512-bit vector of [16 x float] containing the converted values.
static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbf8_ps(__m512 __W, __mmask16 __U, __m128i __A) {
  return (__m512)__builtin_ia32_selectps_512(
      (__mmask16)__U, (__v16sf)_mm512_cvtbf8_ps(__A), (__v16sf)__W);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 512-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82PS instruction.
///
/// \param __U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing BF8 values.
/// \returns
///    A 512-bit vector of [16 x float] containing the converted values.
static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbf8_ps(__mmask16 __U, __m128i __A) {
  return (__m512)__builtin_ia32_selectps_512((__mmask16)__U,
                                             (__v16sf)_mm512_cvtbf8_ps(__A),
                                             (__v16sf)_mm512_setzero_ps());
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 512-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 512-bit vector of [16 x float] containing the converted values.
static __inline__ __m512 __DEFAULT_FN_ATTRS512 _mm512_cvthf8_ps(__m128i __A) {
  return (__m512)__builtin_ia32_vcvthf82ps_512((__v16qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 512-bit vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __W
///    A 512-bit vector of [16 x float] used for writemask.
/// \param __U
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 512-bit vector of [16 x float] containing the converted values.
static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_mask_cvthf8_ps(__m512 __W, __mmask16 __U, __m128i __A) {
  return (__m512)__builtin_ia32_selectps_512(
      (__mmask16)__U, (__v16sf)_mm512_cvthf8_ps(__A), (__v16sf)__W);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    single-precision (32-bit) floating-point elements, and store the results
///    in a 512-bit vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82PS instruction.
///
/// \param __U
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 128-bit vector of [16 x i8] containing HF8 values.
/// \returns
///    A 512-bit vector of [16 x float] containing the converted values.
static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_maskz_cvthf8_ps(__mmask16 __U, __m128i __A) {
  return (__m512)__builtin_ia32_selectps_512((__mmask16)__U,
                                             (__v16sf)_mm512_cvthf8_ps(__A),
                                             (__v16sf)_mm512_setzero_ps());
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    BF6 (6-bit) floating-point elements with saturation, and store the
///    results in a 512-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82BF6S instruction.
///
/// \param __A
///    A 512-bit vector of [64 x i8] containing BF8 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted BF6 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_cvtbf8_bf6s(__m512i __A) {
  return (__m512i)__builtin_ia32_vcvtbf82bf6s_512((__v64qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    HF6 (6-bit) floating-point elements with saturation, and store the
///    results in a 512-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82HF6S instruction.
///
/// \param __A
///    A 512-bit vector of [64 x i8] containing HF8 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF6 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_cvthf8_hf6s(__m512i __A) {
  return (__m512i)__builtin_ia32_vcvthf82hf6s_512((__v64qi)__A);
}

/// Convert packed BF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results in a 256-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF82BF4S instruction.
///
/// \param __A
///    A 512-bit vector of [64 x i8] containing BF8 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted BF4 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS512
_mm512_cvtbf8_bf4s(__m512i __A) {
  return (__m256i)__builtin_ia32_vcvtbf82bf4s_512((__v64qi)__A);
}

/// Convert packed HF8 (8-bit) floating-point elements in \a __A to packed
///    BF4 (4-bit) floating-point elements with saturation, and store the
///    results in a 256-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF82BF4S instruction.
///
/// \param __A
///    A 512-bit vector of [64 x i8] containing HF8 values.
/// \returns
///    A 256-bit vector of [32 x i8] containing the converted BF4 values.
static __inline__ __m256i __DEFAULT_FN_ATTRS512
_mm512_cvthf8_bf4s(__m512i __A) {
  return (__m256i)__builtin_ia32_vcvthf82bf4s_512((__v64qi)__A);
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
///    A pointer to a 256-bit memory location. The address does not need to be
///    aligned.
/// \param __A
///    A 512-bit vector of [64 x i8] containing BF8 values.
static __inline__ void __DEFAULT_FN_ATTRS512
_mm512_cvtbf8_bf4s_storeu(void *__P, __m512i __A) {
  __builtin_ia32_vcvtbf82bf4s_512_mem(__P, (__v64qi)__A);
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
///    A pointer to a 256-bit memory location. The address does not need to be
///    aligned.
/// \param __A
///    A 512-bit vector of [64 x i8] containing HF8 values.
static __inline__ void __DEFAULT_FN_ATTRS512
_mm512_cvthf8_bf4s_storeu(void *__P, __m512i __A) {
  __builtin_ia32_vcvthf82bf4s_512_mem(__P, (__v64qi)__A);
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF4 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512 _mm512_cvtbf4_hf8(__m256i __A) {
  return (__m512i)__builtin_ia32_vcvtbf42hf8_512((__v32qi)__A);
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __W
///    A 512-bit vector of [64 x i8] used for writemask.
/// \param __U
///    A 64-bit mask indicating which elements to write.
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF4 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbf4_hf8(__m512i __W, __mmask64 __U, __m256i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvtbf4_hf8(__A), (__v64qi)__W);
}

/// Convert packed BF4 (4-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF42HF8 instruction.
///
/// \param __U
///    A 64-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 256-bit vector of [32 x i8] containing BF4 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbf4_hf8(__mmask64 __U, __m256i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvtbf4_hf8(__A), (__v64qi)_mm512_setzero_si512());
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __A
///    A 512-bit vector of [64 x i8] containing BF6 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512 _mm512_cvtbf6_hf8(__m512i __A) {
  return (__m512i)__builtin_ia32_vcvtbf62hf8_512((__v64qi)__A);
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __W
///    A 512-bit vector of [64 x i8] used for writemask.
/// \param __U
///    A 64-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [64 x i8] containing BF6 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbf6_hf8(__m512i __W, __mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvtbf6_hf8(__A), (__v64qi)__W);
}

/// Convert packed BF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTBF62HF8 instruction.
///
/// \param __U
///    A 64-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [64 x i8] containing BF6 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbf6_hf8(__mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvtbf6_hf8(__A), (__v64qi)_mm512_setzero_si512());
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __A
///    A 512-bit vector of [64 x i8] containing HF6 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512 _mm512_cvthf6_hf8(__m512i __A) {
  return (__m512i)__builtin_ia32_vcvthf62hf8_512((__v64qi)__A);
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector using writemask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __W
///    A 512-bit vector of [64 x i8] used for writemask.
/// \param __U
///    A 64-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [64 x i8] containing HF6 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_cvthf6_hf8(__m512i __W, __mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvthf6_hf8(__A), (__v64qi)__W);
}

/// Convert packed HF6 (6-bit) floating-point elements in \a __A to packed
///    HF8 (8-bit) floating-point elements, and store the results in a 512-bit
///    vector using zeromask \a __U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VCVTHF62HF8 instruction.
///
/// \param __U
///    A 64-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [64 x i8] containing HF6 values.
/// \returns
///    A 512-bit vector of [64 x i8] containing the converted HF8 values.
static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvthf6_hf8(__mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvthf6_hf8(__A), (__v64qi)_mm512_setzero_si512());
}

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 512-bit vector.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param A
///    A 512-bit vector of [64 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 512-bit vector of [64 x i8] containing the unpacked values.
#define _mm512_unpackb_epi8(A, imm)                                            \
  ((__m512i)__builtin_ia32_vunpackb512((__v64qi)(__m512i)(A), (int)(imm)))

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 512-bit vector using writemask \a U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param W
///    A 512-bit vector of [64 x i8] used for writemask.
/// \param U
///    A 64-bit mask indicating which elements to write.
/// \param A
///    A 512-bit vector of [64 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 512-bit vector of [64 x i8] containing the unpacked values.
#define _mm512_mask_unpackb_epi8(W, U, A, imm)                                 \
  ((__m512i)__builtin_ia32_selectb_512(                                        \
      (__mmask64)(U), (__v64qi)_mm512_unpackb_epi8((A), (imm)),                \
      (__v64qi)(__m512i)(W)))

/// Unpack bytes from \a A according to the immediate value \a imm, and store
///    the results in a 512-bit vector using zeromask \a U.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c VUNPACKB instruction.
///
/// \param U
///    A 64-bit mask indicating which elements to write (zero otherwise).
/// \param A
///    A 512-bit vector of [64 x i8].
/// \param imm
///    An immediate value specifying the unpack operation.
/// \returns
///    A 512-bit vector of [64 x i8] containing the unpacked values.
#define _mm512_maskz_unpackb_epi8(U, A, imm)                                   \
  ((__m512i)__builtin_ia32_selectb_512(                                        \
      (__mmask64)(U), (__v64qi)_mm512_unpackb_epi8((A), (imm)),                \
      (__v64qi)_mm512_setzero_si512()))

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
///    A 512-bit vector of [16 x i32].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvtss_epi32_epi8(__m512i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb512_mask(
      (__v16si)__A, (__v16qi)_mm_setzero_si128(), (__mmask16)-1);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [16 x i32].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtss_epi32_epi8(__m128i __W, __mmask16 __U, __m512i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb512_mask((__v16si)__A, (__v16qi)__W,
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
///    A 16-bit mask indicating which elements to write (zero otherwise).
/// \param __A
///    A 512-bit vector of [16 x i32].
/// \returns
///    A 128-bit vector of [16 x i8] containing the converted values.
static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtss_epi32_epi8(__mmask16 __U, __m512i __A) {
  return (__m128i)__builtin_ia32_vpmovssdb512_mask(
      (__v16si)__A, (__v16qi)_mm_setzero_si128(), __U);
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
///    A 16-bit mask indicating which elements to write.
/// \param __A
///    A 512-bit vector of [16 x i32].
static __inline__ void __DEFAULT_FN_ATTRS512
_mm512_mask_cvtss_epi32_storeu_epi8(void *__P, __mmask16 __M, __m512i __A) {
  __builtin_ia32_vpmovssdb512mem_mask((__v16qi *)__P, (__v16si)__A, __M);
}

#undef __DEFAULT_FN_ATTRS512

#endif // __AVX10_2_512V2AUXINTRIN_H
#endif // __SSE2__
