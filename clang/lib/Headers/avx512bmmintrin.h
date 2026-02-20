/*===-------- avx512bmmintrin.h - AVX512BMM intrinsics *------------------===
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===---------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#error "Never use <avx512bmmintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef _AVX512BMMINTRIN_H
#define _AVX512BMMINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("avx512bmm"),      \
                 __min_vector_width__(512)))

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS_CONSTEXPR __DEFAULT_FN_ATTRS constexpr
#else
#define __DEFAULT_FN_ATTRS_CONSTEXPR __DEFAULT_FN_ATTRS
#endif

/// Multiplies two 16x16 bit matrices using OR reduction and ORs the product
/// into a third 16x16 bit matrix (which is also the destination).
///
/// For the 512-bit ZMM form, each register contains two 16x16 (256-bit)
/// matrices in bits [255:0] and [511:256]. The operation performs:
/// \code{.operation}
///   for i in 0 to 15
///     for j in 0 to 15
///       reduction_bit = __C[16*i+j]
///       for k in 0 to 15
///         reduction_bit |= __A[16*i+k] & __B[16*k+j]
///       end for k
///       dest[16*i+j] = reduction_bit
///     end for j
///   end for i
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> VBMACOR16X16X16 </c> instruction.
///
/// \param __A
///    A 512-bit vector containing two 16x16 bit matrices (one per 256-bit
///    lane).
/// \param __B
///    A 512-bit vector containing two 16x16 bit matrices (one per 256-bit
///    lane).
/// \param __C
///    A 512-bit accumulator vector containing the initial values to OR with.
/// \returns A 512-bit vector containing the accumulated result for each lane.
/// \note This instruction does not support masking.
static __inline __m512i __DEFAULT_FN_ATTRS _mm512_bmacor16x16x16(__m512i __A,
                                                                 __m512i __B,
                                                                 __m512i __C) {
  return (__m512i)__builtin_ia32_bmacor16x16x16_v32hi(
      (__v32hi)__A, (__v32hi)__B, (__v32hi)__C);
}

/// Multiplies two 16x16 bit matrices using XOR reduction and XORs the product
/// into a third 16x16 bit matrix (which is also the destination).
///
/// For the 512-bit ZMM form, each register contains two 16x16 (256-bit)
/// matrices in bits [255:0] and [511:256]. The operation performs:
/// \code{.operation}
///   for i in 0 to 15
///     for j in 0 to 15
///       reduction_bit = __C[16*i+j]
///       for k in 0 to 15
///         reduction_bit ^= __A[16*i+k] & __B[16*k+j]
///       end for k
///       dest[16*i+j] = reduction_bit
///     end for j
///   end for i
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> VBMACXOR16X16X16 </c> instruction.
///
/// \param __A
///    A 512-bit vector containing two 16x16 bit matrices (one per 256-bit
///    lane).
/// \param __B
///    A 512-bit vector containing two 16x16 bit matrices (one per 256-bit
///    lane).
/// \param __C
///    A 512-bit accumulator vector containing the initial values to XOR with.
/// \returns A 512-bit vector containing the accumulated result for each lane.
/// \note This instruction does not support masking.
static __inline __m512i __DEFAULT_FN_ATTRS _mm512_bmacxor16x16x16(__m512i __A,
                                                                  __m512i __B,
                                                                  __m512i __C) {
  return (__m512i)__builtin_ia32_bmacxor16x16x16_v32hi(
      (__v32hi)__A, (__v32hi)__B, (__v32hi)__C);
}

/// Reverses the bits within each byte of the source vector.
///
/// For each byte in the source, reverses the order of its 8 bits to generate
/// the corresponding destination byte. For example, 0b10110001 becomes
/// 0b10001101.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> VBITREV </c> instruction.
///
/// \param __A
///    A 512-bit vector of [64 x i8] where each byte will have its bits
///    reversed.
/// \returns A 512-bit vector of [64 x i8] with bit-reversed bytes.
static __inline __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_bitrev_epi8(__m512i __A) {
  return (__m512i)__builtin_elementwise_bitreverse((__v64qi)__A);
}

/// Reverses the bits within each byte of the source vector, using a writemask
/// to conditionally select elements.
///
/// For each byte position, if the corresponding mask bit is 1, the byte from
/// \a A has its bits reversed and stored in the result. If the mask bit is 0,
/// the corresponding byte from \a B is copied to the result (merge masking).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> VBITREV </c> instruction.
///
/// \param __U
///    A 64-bit mask value where each bit controls one byte (per 8-bit element).
///    A 1 performs bit reversal; a 0 selects the passthrough byte from __B.
/// \param __A
///    A 512-bit vector of [64 x i8] to be bit-reversed.
/// \param __B
///    A 512-bit vector of [64 x i8] providing passthrough values.
/// \returns A 512-bit vector combining bit-reversed and passthrough bytes.
static __inline __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_mask_bitrev_epi8(__mmask64 __U, __m512i __A, __m512i __B) {
  return (__m512i)__builtin_ia32_selectb_512(
      (__mmask64)__U, (__v64qi)_mm512_bitrev_epi8(__A), (__v64qi)__B);
}

/// Reverses the bits within each byte of the source vector, zeroing elements
/// based on the writemask.
///
/// For each byte position, if the corresponding mask bit is 1, the byte from
/// \a A has its bits reversed and stored in the result. If the mask bit is 0,
/// the result byte is set to zero (zero masking).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> VBITREV </c> instruction.
///
/// \param __U
///    A 64-bit mask value where each bit controls one byte (per 8-bit element).
///    A 1 performs bit reversal; a 0 sets the byte to zero.
/// \param __A
///    A 512-bit vector of [64 x i8] to be bit-reversed.
/// \returns A 512-bit vector with bit-reversed or zeroed bytes.
static __inline __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_maskz_bitrev_epi8(__mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512((__mmask64)__U,
                                             (__v64qi)_mm512_bitrev_epi8(__A),
                                             (__v64qi)_mm512_setzero_si512());
}

#undef __DEFAULT_FN_ATTRS
#undef __DEFAULT_FN_ATTRS_CONSTEXPR

#endif
