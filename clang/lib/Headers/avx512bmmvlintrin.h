/*===------------- avx512bmvlintrin.h - BMM intrinsics ------------------===
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error                                                                         \
    "Never use <avx512bmmvlintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __BMMVLINTRIN_H
#define __BMMVLINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS128                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512bmm,avx512vl"), __min_vector_width__(128)))
#define __DEFAULT_FN_ATTRS256                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512bmm,avx512vl"), __min_vector_width__(256)))

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS128_CONSTEXPR __DEFAULT_FN_ATTRS128 constexpr
#define __DEFAULT_FN_ATTRS256_CONSTEXPR __DEFAULT_FN_ATTRS256 constexpr
#else
#define __DEFAULT_FN_ATTRS128_CONSTEXPR __DEFAULT_FN_ATTRS128
#define __DEFAULT_FN_ATTRS256_CONSTEXPR __DEFAULT_FN_ATTRS256
#endif

/// Multiplies two 16x16 bit matrices using OR reduction and ORs the product
/// into a third 16x16 bit matrix (which is also the destination).
///
/// For the 256-bit YMM form, the source registers/memory each contain a single
/// 16x16 (256-bit) matrix in bits [255:0]. The operation performs:
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
///    A 256-bit vector containing a 16x16 bit matrix.
/// \param __B
///    A 256-bit vector containing a 16x16 bit matrix.
/// \param __C
///    A 256-bit accumulator vector containing the initial values to OR with.
/// \returns A 256-bit vector containing the accumulated result.
/// \note This instruction does not support masking.
static __inline __m256i __DEFAULT_FN_ATTRS256
_mm256_bmacor16x16x16(__m256i __A, __m256i __B, __m256i __C) {
  return (__m256i)__builtin_ia32_bmacor16x16x16_v16hi(
      (__v16hi)__A, (__v16hi)__B, (__v16hi)__C);
}

/// Multiplies two 16x16 bit matrices using XOR reduction and XORs the product
/// into a third 16x16 bit matrix (which is also the destination).
///
/// For the 256-bit YMM form, the source registers/memory each contain a single
/// 16x16 (256-bit) matrix in bits [255:0]. The operation performs:
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
///    A 256-bit vector containing a 16x16 bit matrix.
/// \param __B
///    A 256-bit vector containing a 16x16 bit matrix.
/// \param __C
///    A 256-bit accumulator vector containing the initial values to XOR with.
/// \returns A 256-bit vector containing the accumulated result.
/// \note This instruction does not support masking.
static __inline __m256i __DEFAULT_FN_ATTRS256
_mm256_bmacxor16x16x16(__m256i __A, __m256i __B, __m256i __C) {
  return (__m256i)__builtin_ia32_bmacxor16x16x16_v16hi(
      (__v16hi)__A, (__v16hi)__B, (__v16hi)__C);
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
///    A 128-bit vector of [16 x i8] where each byte will have its bits
///    reversed.
/// \returns A 128-bit vector of [16 x i8] with bit-reversed bytes.
static __inline __m128i __DEFAULT_FN_ATTRS128_CONSTEXPR
_mm128_bitrev_epi8(__m128i __A) {
  return (__m128i)__builtin_elementwise_bitreverse((__v16qi)__A);
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
///    A 256-bit vector of [32 x i8] where each byte will have its bits
///    reversed.
/// \returns A 256-bit vector of [32 x i8] with bit-reversed bytes.
static __inline __m256i __DEFAULT_FN_ATTRS256_CONSTEXPR
_mm256_bitrev_epi8(__m256i __A) {
  return (__m256i)__builtin_elementwise_bitreverse((__v32qi)__A);
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
///    A 16-bit mask value where each bit controls one byte (per 8-bit element).
///    A 1 performs bit reversal; a 0 selects the passthrough byte from __B.
/// \param __A
///    A 128-bit vector of [16 x i8] to be bit-reversed.
/// \param __B
///    A 128-bit vector of [16 x i8] providing passthrough values.
/// \returns A 128-bit vector combining bit-reversed and passthrough bytes.
static __inline __m128i __DEFAULT_FN_ATTRS128_CONSTEXPR
_mm128_mask_bitrev_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  return (__m128i)__builtin_ia32_selectb_128(
      (__mmask16)__U, (__v16qi)_mm128_bitrev_epi8(__A), (__v16qi)__B);
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
///    A 32-bit mask value where each bit controls one byte (per 8-bit element).
///    A 1 performs bit reversal; a 0 selects the passthrough byte from __B.
/// \param __A
///    A 256-bit vector of [32 x i8] to be bit-reversed.
/// \param __B
///    A 256-bit vector of [32 x i8] providing passthrough values.
/// \returns A 256-bit vector combining bit-reversed and passthrough bytes.
static __inline __m256i __DEFAULT_FN_ATTRS256_CONSTEXPR
_mm256_mask_bitrev_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  return (__m256i)__builtin_ia32_selectb_256(
      (__mmask32)__U, (__v32qi)_mm256_bitrev_epi8(__A), (__v32qi)__B);
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
///    A 16-bit mask value where each bit controls one byte (per 8-bit element).
///    A 1 performs bit reversal; a 0 sets the byte to zero.
/// \param __A
///    A 128-bit vector of [16 x i8] to be bit-reversed.
/// \returns A 128-bit vector with bit-reversed or zeroed bytes.
static __inline __m128i __DEFAULT_FN_ATTRS128_CONSTEXPR
_mm128_maskz_bitrev_epi8(__mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128((__mmask16)__U,
                                             (__v16qi)_mm128_bitrev_epi8(__A),
                                             (__v16qi)_mm_setzero_si128());
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
///    A 32-bit mask value where each bit controls one byte (per 8-bit element).
///    A 1 performs bit reversal; a 0 sets the byte to zero.
/// \param __A
///    A 256-bit vector of [32 x i8] to be bit-reversed.
/// \returns A 256-bit vector with bit-reversed or zeroed bytes.
static __inline __m256i __DEFAULT_FN_ATTRS256_CONSTEXPR
_mm256_maskz_bitrev_epi8(__mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256((__mmask32)__U,
                                             (__v32qi)_mm256_bitrev_epi8(__A),
                                             (__v32qi)_mm256_setzero_si256());
}

#undef __DEFAULT_FN_ATTRS128_CONSTEXPR
#undef __DEFAULT_FN_ATTRS256_CONSTEXPR
#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256

#endif
