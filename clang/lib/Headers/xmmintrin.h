/*===---- xmmintrin.h - SSE intrinsics -------------------------------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __XMMINTRIN_H
#define __XMMINTRIN_H

#include <mmintrin.h>

typedef int __v4si __attribute__((__vector_size__(16)));
typedef float __v4sf __attribute__((__vector_size__(16)));
typedef float __m128 __attribute__((__vector_size__(16)));

/* This header should only be included in a hosted environment as it depends on
 * a standard library to provide allocation routines. */
#if __STDC_HOSTED__
#include <mm_malloc.h>
#endif

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("sse")))

/// \brief Adds the 32-bit float values in the low-order bits of the operands.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VADDSS / ADDSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the source operands.
///    The lower 32 bits of this operand are used in the calculation.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the source operands.
///    The lower 32 bits of this operand are used in the calculation.
/// \returns A 128-bit vector of [4 x float] whose lower 32 bits contain the sum
///    of the lower 32 bits of both operands. The upper 96 bits are copied from
///    the upper 96 bits of the first source operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_add_ss(__m128 __a, __m128 __b)
{
  __a[0] += __b[0];
  return __a;
}

/// \brief Adds two 128-bit vectors of [4 x float], and returns the results of
///    the addition.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VADDPS / ADDPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the source operands.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the source operands.
/// \returns A 128-bit vector of [4 x float] containing the sums of both
///    operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_add_ps(__m128 __a, __m128 __b)
{
  return __a + __b;
}

/// \brief Subtracts the 32-bit float value in the low-order bits of the second
///    operand from the corresponding value in the first operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VSUBSS / SUBSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing the minuend. The lower 32 bits
///    of this operand are used in the calculation.
/// \param __b
///    A 128-bit vector of [4 x float] containing the subtrahend. The lower 32
///    bits of this operand are used in the calculation.
/// \returns A 128-bit vector of [4 x float] whose lower 32 bits contain the
///    difference of the lower 32 bits of both operands. The upper 96 bits are
///    copied from the upper 96 bits of the first source operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_sub_ss(__m128 __a, __m128 __b)
{
  __a[0] -= __b[0];
  return __a;
}

/// \brief Subtracts each of the values of the second operand from the first
///    operand, both of which are 128-bit vectors of [4 x float] and returns
///    the results of the subtraction.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VSUBPS / SUBPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing the minuend.
/// \param __b
///    A 128-bit vector of [4 x float] containing the subtrahend.
/// \returns A 128-bit vector of [4 x float] containing the differences between
///    both operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_sub_ps(__m128 __a, __m128 __b)
{
  return __a - __b;
}

/// \brief Multiplies two 32-bit float values in the low-order bits of the
///    operands.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMULSS / MULSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the source operands.
///    The lower 32 bits of this operand are used in the calculation.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the source operands.
///    The lower 32 bits of this operand are used in the calculation.
/// \returns A 128-bit vector of [4 x float] containing the product of the lower
///    32 bits of both operands. The upper 96 bits are copied from the upper 96
///    bits of the first source operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mul_ss(__m128 __a, __m128 __b)
{
  __a[0] *= __b[0];
  return __a;
}

/// \brief Multiplies two 128-bit vectors of [4 x float] and returns the
///    results of the multiplication.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMULPS / MULPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the source operands.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the source operands.
/// \returns A 128-bit vector of [4 x float] containing the products of both
///    operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_mul_ps(__m128 __a, __m128 __b)
{
  return __a * __b;
}

/// \brief Divides the value in the low-order 32 bits of the first operand by
///    the corresponding value in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VDIVSS / DIVSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing the dividend. The lower 32
///    bits of this operand are used in the calculation.
/// \param __b
///    A 128-bit vector of [4 x float] containing the divisor. The lower 32 bits
///    of this operand are used in the calculation.
/// \returns A 128-bit vector of [4 x float] containing the quotients of the
///    lower 32 bits of both operands. The upper 96 bits are copied from the
///    upper 96 bits of the first source operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_div_ss(__m128 __a, __m128 __b)
{
  __a[0] /= __b[0];
  return __a;
}

/// \brief Divides two 128-bit vectors of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VDIVPS / DIVPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing the dividend.
/// \param __b
///    A 128-bit vector of [4 x float] containing the divisor.
/// \returns A 128-bit vector of [4 x float] containing the quotients of both
///    operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_div_ps(__m128 __a, __m128 __b)
{
  return __a / __b;
}

/// \brief Calculates the square root of the value stored in the low-order bits
///    of a 128-bit vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VSQRTSS / SQRTSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the calculation.
/// \returns A 128-bit vector of [4 x float] containing the square root of the
///    value in the low-order bits of the operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_sqrt_ss(__m128 __a)
{
  __m128 __c = __builtin_ia32_sqrtss(__a);
  return (__m128) { __c[0], __a[1], __a[2], __a[3] };
}

/// \brief Calculates the square roots of the values stored in a 128-bit vector
///    of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VSQRTPS / SQRTPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the square roots of the
///    values in the operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_sqrt_ps(__m128 __a)
{
  return __builtin_ia32_sqrtps(__a);
}

/// \brief Calculates the approximate reciprocal of the value stored in the
///    low-order bits of a 128-bit vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VRCPSS / RCPSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the calculation.
/// \returns A 128-bit vector of [4 x float] containing the approximate
///    reciprocal of the value in the low-order bits of the operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_rcp_ss(__m128 __a)
{
  __m128 __c = __builtin_ia32_rcpss(__a);
  return (__m128) { __c[0], __a[1], __a[2], __a[3] };
}

/// \brief Calculates the approximate reciprocals of the values stored in a
///    128-bit vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VRCPPS / RCPPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the approximate
///    reciprocals of the values in the operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_rcp_ps(__m128 __a)
{
  return __builtin_ia32_rcpps(__a);
}

/// \brief Calculates the approximate reciprocal of the square root of the value
///    stored in the low-order bits of a 128-bit vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VRSQRTSS / RSQRTSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the calculation.
/// \returns A 128-bit vector of [4 x float] containing the approximate
///    reciprocal of the square root of the value in the low-order bits of the
///    operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_rsqrt_ss(__m128 __a)
{
  __m128 __c = __builtin_ia32_rsqrtss(__a);
  return (__m128) { __c[0], __a[1], __a[2], __a[3] };
}

/// \brief Calculates the approximate reciprocals of the square roots of the
///    values stored in a 128-bit vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VRSQRTPS / RSQRTPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the approximate
///    reciprocals of the square roots of the values in the operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_rsqrt_ps(__m128 __a)
{
  return __builtin_ia32_rsqrtps(__a);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands and returns the lesser value in the low-order bits of the
///    vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMINSS / MINSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] whose lower 32 bits contain the
///    minimum value between both operands. The upper 96 bits are copied from
///    the upper 96 bits of the first source operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_min_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_minss(__a, __b);
}

/// \brief Compares two 128-bit vectors of [4 x float] and returns the
///    lesser of each pair of values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMINPS / MINPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands.
/// \returns A 128-bit vector of [4 x float] containing the minimum values
///    between both operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_min_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_minps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands and returns the greater value in the low-order bits of
///    a vector [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMAXSS / MAXSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] whose lower 32 bits contain the
///    maximum value between both operands. The upper 96 bits are copied from
///    the upper 96 bits of the first source operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_max_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_maxss(__a, __b);
}

/// \brief Compares two 128-bit vectors of [4 x float] and returns the greater
///    of each pair of values.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VMAXPS / MAXPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands.
/// \returns A 128-bit vector of [4 x float] containing the maximum values
///    between both operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_max_ps(__m128 __a, __m128 __b)
{
  return __builtin_ia32_maxps(__a, __b);
}

/// \brief Performs a bitwise AND of two 128-bit vectors of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VANDPS / ANDPS instructions.
///
/// \param __a
///    A 128-bit vector containing one of the source operands.
/// \param __b
///    A 128-bit vector containing one of the source operands.
/// \returns A 128-bit vector of [4 x float] containing the bitwise AND of the
///    values between both operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_and_ps(__m128 __a, __m128 __b)
{
  return (__m128)((__v4si)__a & (__v4si)__b);
}

/// \brief Performs a bitwise AND of two 128-bit vectors of [4 x float], using
///    the one's complement of the values contained in the first source
///    operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VANDNPS / ANDNPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing the first source operand. The
///    one's complement of this value is used in the bitwise AND.
/// \param __b
///    A 128-bit vector of [4 x float] containing the second source operand.
/// \returns A 128-bit vector of [4 x float] containing the bitwise AND of the
///    one's complement of the first operand and the values in the second
///    operand.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_andnot_ps(__m128 __a, __m128 __b)
{
  return (__m128)(~(__v4si)__a & (__v4si)__b);
}

/// \brief Performs a bitwise OR of two 128-bit vectors of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VORPS / ORPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the source operands.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the source operands.
/// \returns A 128-bit vector of [4 x float] containing the bitwise OR of the
///    values between both operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_or_ps(__m128 __a, __m128 __b)
{
  return (__m128)((__v4si)__a | (__v4si)__b);
}

/// \brief Performs a bitwise exclusive OR of two 128-bit vectors of
///    [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VXORPS / XORPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the source operands.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the source operands.
/// \returns A 128-bit vector of [4 x float] containing the bitwise exclusive OR
///    of the values between both operands.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_xor_ps(__m128 __a, __m128 __b)
{
  return (__m128)((__v4si)__a ^ (__v4si)__b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands for equality and returns the result of the comparison in the
///    low-order bits of a vector [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPEQSS / CMPEQSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpeq_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpeqss(__a, __b);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] for equality.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPEQPS / CMPEQPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpeq_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpeqps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is less than the
///    corresponding value in the second operand and returns the result of the
///    comparison in the low-order bits of a vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPLTSS / CMPLTSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmplt_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpltss(__a, __b);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are less than those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPLTPS / CMPLTPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmplt_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpltps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is less than or
///    equal to the corresponding value in the second operand and returns the
///    result of the comparison in the low-order bits of a vector of
///    [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPLESS / CMPLESS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmple_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpless(__a, __b);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are less than or equal to those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPLEPS / CMPLEPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmple_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpleps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is greater than
///    the corresponding value in the second operand and returns the result of
///    the comparison in the low-order bits of a vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPLTSS / CMPLTSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpgt_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_shufflevector(__a,
                                         __builtin_ia32_cmpltss(__b, __a),
                                         4, 1, 2, 3);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are greater than those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPLTPS / CMPLTPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpgt_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpltps(__b, __a);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is greater than
///    or equal to the corresponding value in the second operand and returns
///    the result of the comparison in the low-order bits of a vector of
///    [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPLESS / CMPLESS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpge_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_shufflevector(__a,
                                         __builtin_ia32_cmpless(__b, __a),
                                         4, 1, 2, 3);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are greater than or equal to those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPLEPS / CMPLEPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpge_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpleps(__b, __a);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands for inequality and returns the result of the comparison in the
///    low-order bits of a vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNEQSS / CMPNEQSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpneq_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpneqss(__a, __b);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] for inequality.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNEQPS / CMPNEQPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpneq_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpneqps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is not less than
///    the corresponding value in the second operand and returns the result of
///    the comparison in the low-order bits of a vector of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNLTSS / CMPNLTSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpnlt_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpnltss(__a, __b);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are not less than those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNLTPS / CMPNLTPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpnlt_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpnltps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is not less than
///    or equal to the corresponding value in the second operand and returns
///    the result of the comparison in the low-order bits of a vector of
///    [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNLESS / CMPNLESS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpnle_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpnless(__a, __b);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are not less than or equal to those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNLEPS / CMPNLEPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpnle_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpnleps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is not greater
///    than the corresponding value in the second operand and returns the
///    result of the comparison in the low-order bits of a vector of
///    [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNLTSS / CMPNLTSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpngt_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_shufflevector(__a,
                                         __builtin_ia32_cmpnltss(__b, __a),
                                         4, 1, 2, 3);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are not greater than those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNLTPS / CMPNLTPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpngt_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpnltps(__b, __a);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is not greater
///    than or equal to the corresponding value in the second operand and
///    returns the result of the comparison in the low-order bits of a vector
///    of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNLESS / CMPNLESS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpnge_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_shufflevector(__a,
                                         __builtin_ia32_cmpnless(__b, __a),
                                         4, 1, 2, 3);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are not greater than or equal to those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPNLEPS / CMPNLEPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpnge_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpnleps(__b, __a);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is ordered with
///    respect to the corresponding value in the second operand and returns the
///    result of the comparison in the low-order bits of a vector of
///    [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPORDSS / CMPORDSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpord_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpordss(__a, __b);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are ordered with respect to those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPORDPS / CMPORDPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpord_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpordps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the value in the first operand is unordered
///    with respect to the corresponding value in the second operand and
///    returns the result of the comparison in the low-order bits of a vector
///    of [4 x float].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPUNORDSS / CMPUNORDSS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float] containing one of the operands. The lower
///    32 bits of this operand are used in the comparison.
/// \returns A 128-bit vector of [4 x float] containing the comparison results
///    in the low-order bits.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpunord_ss(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpunordss(__a, __b);
}

/// \brief Compares each of the corresponding 32-bit float values of the
///    128-bit vectors of [4 x float] to determine if the values in the first
///    operand are unordered with respect to those in the second operand.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCMPUNORDPS / CMPUNORDPS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \param __b
///    A 128-bit vector of [4 x float].
/// \returns A 128-bit vector of [4 x float] containing the comparison results.
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cmpunord_ps(__m128 __a, __m128 __b)
{
  return (__m128)__builtin_ia32_cmpunordps(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands for equality and returns the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCOMISS / COMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_comieq_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comieq(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the first operand is less than the second
///    operand and returns the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCOMISS / COMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_comilt_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comilt(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the first operand is less than or equal to the
///    second operand and returns the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCOMISS / COMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_comile_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comile(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the first operand is greater than the second
///    operand and returns the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCOMISS / COMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_comigt_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comigt(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the first operand is greater than or equal to
///    the second operand and returns the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCOMISS / COMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_comige_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comige(__a, __b);
}

/// \brief Compares two 32-bit float values in the low-order bits of both
///    operands to determine if the first operand is not equal to the second
///    operand and returns the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCOMISS / COMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_comineq_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_comineq(__a, __b);
}

/// \brief Performs an unordered comparison of two 32-bit float values using
///    the low-order bits of both operands to determine equality and returns
///    the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VUCOMISS / UCOMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomieq_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomieq(__a, __b);
}

/// \brief Performs an unordered comparison of two 32-bit float values using
///    the low-order bits of both operands to determine if the first operand is
///    less than the second operand and returns the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VUCOMISS / UCOMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomilt_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomilt(__a, __b);
}

/// \brief Performs an unordered comparison of two 32-bit float values using
///    the low-order bits of both operands to determine if the first operand
///    is less than or equal to the second operand and returns the result of
///    the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VUCOMISS / UCOMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomile_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomile(__a, __b);
}

/// \brief Performs an unordered comparison of two 32-bit float values using
///    the low-order bits of both operands to determine if the first operand
///    is greater than the second operand and returns the result of the
///    comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VUCOMISS / UCOMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomigt_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomigt(__a, __b);
}

/// \brief Performs an unordered comparison of two 32-bit float values using
///    the low-order bits of both operands to determine if the first operand is
///    greater than or equal to the second operand and returns the result of
///    the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VUCOMISS / UCOMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomige_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomige(__a, __b);
}

/// \brief Performs an unordered comparison of two 32-bit float values using
///    the low-order bits of both operands to determine inequality and returns
///    the result of the comparison.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VUCOMISS / UCOMISS instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \param __b
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the comparison.
/// \returns An integer containing the comparison results.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_ucomineq_ss(__m128 __a, __m128 __b)
{
  return __builtin_ia32_ucomineq(__a, __b);
}

/// \brief Converts a float value contained in the lower 32 bits of a vector of
///    [4 x float] into a 32-bit integer.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTSS2SI / CVTSS2SI instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the conversion.
/// \returns A 32-bit integer containing the converted value.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvtss_si32(__m128 __a)
{
  return __builtin_ia32_cvtss2si(__a);
}

/// \brief Converts a float value contained in the lower 32 bits of a vector of
///    [4 x float] into a 32-bit integer.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTSS2SI / CVTSS2SI instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the conversion.
/// \returns A 32-bit integer containing the converted value.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvt_ss2si(__m128 __a)
{
  return _mm_cvtss_si32(__a);
}

#ifdef __x86_64__

/// \brief Converts a float value contained in the lower 32 bits of a vector of
///    [4 x float] into a 64-bit integer.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTSS2SI / CVTSS2SI instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the conversion.
/// \returns A 64-bit integer containing the converted value.
static __inline__ long long __DEFAULT_FN_ATTRS
_mm_cvtss_si64(__m128 __a)
{
  return __builtin_ia32_cvtss2si64(__a);
}

#endif

/// \brief Converts two low-order float values in a 128-bit vector of
///    [4 x float] into a 64-bit vector of [2 x i32].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c CVTPS2PI instruction.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \returns A 64-bit integer vector containing the converted values.
static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvtps_pi32(__m128 __a)
{
  return (__m64)__builtin_ia32_cvtps2pi(__a);
}

/// \brief Converts two low-order float values in a 128-bit vector of
///    [4 x float] into a 64-bit vector of [2 x i32].
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c CVTPS2PI instruction.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \returns A 64-bit integer vector containing the converted values.
static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvt_ps2pi(__m128 __a)
{
  return _mm_cvtps_pi32(__a);
}

/// \brief Converts a float value contained in the lower 32 bits of a vector of
///    [4 x float] into a 32-bit integer, truncating the result when it is
///    inexact.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTTSS2SI / CVTTSS2SI instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the conversion.
/// \returns A 32-bit integer containing the converted value.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvttss_si32(__m128 __a)
{
  return __a[0];
}

/// \brief Converts a float value contained in the lower 32 bits of a vector of
///    [4 x float] into a 32-bit integer, truncating the result when it is
///    inexact.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTTSS2SI / CVTTSS2SI instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the conversion.
/// \returns A 32-bit integer containing the converted value.
static __inline__ int __DEFAULT_FN_ATTRS
_mm_cvtt_ss2si(__m128 __a)
{
  return _mm_cvttss_si32(__a);
}

/// \brief Converts a float value contained in the lower 32 bits of a vector of
///    [4 x float] into a 64-bit integer, truncating the result when it is
///    inexact.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c VCVTTSS2SI / CVTTSS2SI instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float]. The lower 32 bits of this operand are
///    used in the conversion.
/// \returns A 64-bit integer containing the converted value.
static __inline__ long long __DEFAULT_FN_ATTRS
_mm_cvttss_si64(__m128 __a)
{
  return __a[0];
}

/// \brief Converts two low-order float values in a 128-bit vector of
///    [4 x float] into a 64-bit vector of [2 x i32], truncating the result
///    when it is inexact.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c CVTTPS2PI / VTTPS2PI instructions.
///
/// \param __a
///    A 128-bit vector of [4 x float].
/// \returns A 64-bit integer vector containing the converted values.
static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvttps_pi32(__m128 __a)
{
  return (__m64)__builtin_ia32_cvttps2pi(__a);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvtt_ps2pi(__m128 __a)
{
  return _mm_cvttps_pi32(__a);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtsi32_ss(__m128 __a, int __b)
{
  __a[0] = __b;
  return __a;
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvt_si2ss(__m128 __a, int __b)
{
  return _mm_cvtsi32_ss(__a, __b);
}

#ifdef __x86_64__

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtsi64_ss(__m128 __a, long long __b)
{
  __a[0] = __b;
  return __a;
}

#endif

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtpi32_ps(__m128 __a, __m64 __b)
{
  return __builtin_ia32_cvtpi2ps(__a, (__v2si)__b);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvt_pi2ps(__m128 __a, __m64 __b)
{
  return _mm_cvtpi32_ps(__a, __b);
}

static __inline__ float __DEFAULT_FN_ATTRS
_mm_cvtss_f32(__m128 __a)
{
  return __a[0];
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_loadh_pi(__m128 __a, const __m64 *__p)
{
  typedef float __mm_loadh_pi_v2f32 __attribute__((__vector_size__(8)));
  struct __mm_loadh_pi_struct {
    __mm_loadh_pi_v2f32 __u;
  } __attribute__((__packed__, __may_alias__));
  __mm_loadh_pi_v2f32 __b = ((struct __mm_loadh_pi_struct*)__p)->__u;
  __m128 __bb = __builtin_shufflevector(__b, __b, 0, 1, 0, 1);
  return __builtin_shufflevector(__a, __bb, 0, 1, 4, 5);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_loadl_pi(__m128 __a, const __m64 *__p)
{
  typedef float __mm_loadl_pi_v2f32 __attribute__((__vector_size__(8)));
  struct __mm_loadl_pi_struct {
    __mm_loadl_pi_v2f32 __u;
  } __attribute__((__packed__, __may_alias__));
  __mm_loadl_pi_v2f32 __b = ((struct __mm_loadl_pi_struct*)__p)->__u;
  __m128 __bb = __builtin_shufflevector(__b, __b, 0, 1, 0, 1);
  return __builtin_shufflevector(__a, __bb, 4, 5, 2, 3);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_load_ss(const float *__p)
{
  struct __mm_load_ss_struct {
    float __u;
  } __attribute__((__packed__, __may_alias__));
  float __u = ((struct __mm_load_ss_struct*)__p)->__u;
  return (__m128){ __u, 0, 0, 0 };
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_load1_ps(const float *__p)
{
  struct __mm_load1_ps_struct {
    float __u;
  } __attribute__((__packed__, __may_alias__));
  float __u = ((struct __mm_load1_ps_struct*)__p)->__u;
  return (__m128){ __u, __u, __u, __u };
}

#define        _mm_load_ps1(p) _mm_load1_ps(p)

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_load_ps(const float *__p)
{
  return *(__m128*)__p;
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_loadu_ps(const float *__p)
{
  struct __loadu_ps {
    __m128 __v;
  } __attribute__((__packed__, __may_alias__));
  return ((struct __loadu_ps*)__p)->__v;
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_loadr_ps(const float *__p)
{
  __m128 __a = _mm_load_ps(__p);
  return __builtin_shufflevector(__a, __a, 3, 2, 1, 0);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_undefined_ps()
{
  return (__m128)__builtin_ia32_undef128();
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_set_ss(float __w)
{
  return (__m128){ __w, 0, 0, 0 };
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_set1_ps(float __w)
{
  return (__m128){ __w, __w, __w, __w };
}

/* Microsoft specific. */
static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_set_ps1(float __w)
{
    return _mm_set1_ps(__w);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_set_ps(float __z, float __y, float __x, float __w)
{
  return (__m128){ __w, __x, __y, __z };
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_setr_ps(float __z, float __y, float __x, float __w)
{
  return (__m128){ __z, __y, __x, __w };
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_setzero_ps(void)
{
  return (__m128){ 0, 0, 0, 0 };
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storeh_pi(__m64 *__p, __m128 __a)
{
  __builtin_ia32_storehps((__v2si *)__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storel_pi(__m64 *__p, __m128 __a)
{
  __builtin_ia32_storelps((__v2si *)__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store_ss(float *__p, __m128 __a)
{
  struct __mm_store_ss_struct {
    float __u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_store_ss_struct*)__p)->__u = __a[0];
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storeu_ps(float *__p, __m128 __a)
{
  __builtin_ia32_storeups(__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store1_ps(float *__p, __m128 __a)
{
  __a = __builtin_shufflevector(__a, __a, 0, 0, 0, 0);
  _mm_storeu_ps(__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store_ps1(float *__p, __m128 __a)
{
    return _mm_store1_ps(__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_store_ps(float *__p, __m128 __a)
{
  *(__m128 *)__p = __a;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storer_ps(float *__p, __m128 __a)
{
  __a = __builtin_shufflevector(__a, __a, 3, 2, 1, 0);
  _mm_store_ps(__p, __a);
}

#define _MM_HINT_T0 3
#define _MM_HINT_T1 2
#define _MM_HINT_T2 1
#define _MM_HINT_NTA 0

#ifndef _MSC_VER
/* FIXME: We have to #define this because "sel" must be a constant integer, and
   Sema doesn't do any form of constant propagation yet. */

#define _mm_prefetch(a, sel) (__builtin_prefetch((void *)(a), 0, (sel)))
#endif

static __inline__ void __DEFAULT_FN_ATTRS
_mm_stream_pi(__m64 *__p, __m64 __a)
{
  __builtin_ia32_movntq(__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_stream_ps(float *__p, __m128 __a)
{
  __builtin_ia32_movntps(__p, __a);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_sfence(void)
{
  __builtin_ia32_sfence();
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_extract_pi16(__m64 __a, int __n)
{
  __v4hi __b = (__v4hi)__a;
  return (unsigned short)__b[__n & 3];
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_insert_pi16(__m64 __a, int __d, int __n)
{
   __v4hi __b = (__v4hi)__a;
   __b[__n & 3] = __d;
   return (__m64)__b;
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_max_pi16(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pmaxsw((__v4hi)__a, (__v4hi)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_max_pu8(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pmaxub((__v8qi)__a, (__v8qi)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_min_pi16(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pminsw((__v4hi)__a, (__v4hi)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_min_pu8(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pminub((__v8qi)__a, (__v8qi)__b);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_movemask_pi8(__m64 __a)
{
  return __builtin_ia32_pmovmskb((__v8qi)__a);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_mulhi_pu16(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pmulhuw((__v4hi)__a, (__v4hi)__b);
}

#define _mm_shuffle_pi16(a, n) __extension__ ({ \
  (__m64)__builtin_ia32_pshufw((__v4hi)(__m64)(a), (n)); })

static __inline__ void __DEFAULT_FN_ATTRS
_mm_maskmove_si64(__m64 __d, __m64 __n, char *__p)
{
  __builtin_ia32_maskmovq((__v8qi)__d, (__v8qi)__n, __p);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_avg_pu8(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pavgb((__v8qi)__a, (__v8qi)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_avg_pu16(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_pavgw((__v4hi)__a, (__v4hi)__b);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_sad_pu8(__m64 __a, __m64 __b)
{
  return (__m64)__builtin_ia32_psadbw((__v8qi)__a, (__v8qi)__b);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
_mm_getcsr(void)
{
  return __builtin_ia32_stmxcsr();
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_setcsr(unsigned int __i)
{
  __builtin_ia32_ldmxcsr(__i);
}

#define _mm_shuffle_ps(a, b, mask) __extension__ ({ \
  (__m128)__builtin_shufflevector((__v4sf)(__m128)(a), (__v4sf)(__m128)(b), \
                                  (mask) & 0x3, ((mask) & 0xc) >> 2, \
                                  (((mask) & 0x30) >> 4) + 4, \
                                  (((mask) & 0xc0) >> 6) + 4); })

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_unpackhi_ps(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 2, 6, 3, 7);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_unpacklo_ps(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 0, 4, 1, 5);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_move_ss(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 4, 1, 2, 3);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_movehl_ps(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 6, 7, 2, 3);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_movelh_ps(__m128 __a, __m128 __b)
{
  return __builtin_shufflevector(__a, __b, 0, 1, 4, 5);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtpi16_ps(__m64 __a)
{
  __m64 __b, __c;
  __m128 __r;

  __b = _mm_setzero_si64();
  __b = _mm_cmpgt_pi16(__b, __a);
  __c = _mm_unpackhi_pi16(__a, __b);
  __r = _mm_setzero_ps();
  __r = _mm_cvtpi32_ps(__r, __c);
  __r = _mm_movelh_ps(__r, __r);
  __c = _mm_unpacklo_pi16(__a, __b);
  __r = _mm_cvtpi32_ps(__r, __c);

  return __r;
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtpu16_ps(__m64 __a)
{
  __m64 __b, __c;
  __m128 __r;

  __b = _mm_setzero_si64();
  __c = _mm_unpackhi_pi16(__a, __b);
  __r = _mm_setzero_ps();
  __r = _mm_cvtpi32_ps(__r, __c);
  __r = _mm_movelh_ps(__r, __r);
  __c = _mm_unpacklo_pi16(__a, __b);
  __r = _mm_cvtpi32_ps(__r, __c);

  return __r;
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtpi8_ps(__m64 __a)
{
  __m64 __b;

  __b = _mm_setzero_si64();
  __b = _mm_cmpgt_pi8(__b, __a);
  __b = _mm_unpacklo_pi8(__a, __b);

  return _mm_cvtpi16_ps(__b);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtpu8_ps(__m64 __a)
{
  __m64 __b;

  __b = _mm_setzero_si64();
  __b = _mm_unpacklo_pi8(__a, __b);

  return _mm_cvtpi16_ps(__b);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_cvtpi32x2_ps(__m64 __a, __m64 __b)
{
  __m128 __c;

  __c = _mm_setzero_ps();
  __c = _mm_cvtpi32_ps(__c, __b);
  __c = _mm_movelh_ps(__c, __c);

  return _mm_cvtpi32_ps(__c, __a);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvtps_pi16(__m128 __a)
{
  __m64 __b, __c;

  __b = _mm_cvtps_pi32(__a);
  __a = _mm_movehl_ps(__a, __a);
  __c = _mm_cvtps_pi32(__a);

  return _mm_packs_pi32(__b, __c);
}

static __inline__ __m64 __DEFAULT_FN_ATTRS
_mm_cvtps_pi8(__m128 __a)
{
  __m64 __b, __c;

  __b = _mm_cvtps_pi16(__a);
  __c = _mm_setzero_si64();

  return _mm_packs_pi16(__b, __c);
}

static __inline__ int __DEFAULT_FN_ATTRS
_mm_movemask_ps(__m128 __a)
{
  return __builtin_ia32_movmskps(__a);
}


#ifdef _MSC_VER
#define _MM_ALIGN16 __declspec(align(16))
#endif

#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

#define _MM_EXCEPT_INVALID    (0x0001)
#define _MM_EXCEPT_DENORM     (0x0002)
#define _MM_EXCEPT_DIV_ZERO   (0x0004)
#define _MM_EXCEPT_OVERFLOW   (0x0008)
#define _MM_EXCEPT_UNDERFLOW  (0x0010)
#define _MM_EXCEPT_INEXACT    (0x0020)
#define _MM_EXCEPT_MASK       (0x003f)

#define _MM_MASK_INVALID      (0x0080)
#define _MM_MASK_DENORM       (0x0100)
#define _MM_MASK_DIV_ZERO     (0x0200)
#define _MM_MASK_OVERFLOW     (0x0400)
#define _MM_MASK_UNDERFLOW    (0x0800)
#define _MM_MASK_INEXACT      (0x1000)
#define _MM_MASK_MASK         (0x1f80)

#define _MM_ROUND_NEAREST     (0x0000)
#define _MM_ROUND_DOWN        (0x2000)
#define _MM_ROUND_UP          (0x4000)
#define _MM_ROUND_TOWARD_ZERO (0x6000)
#define _MM_ROUND_MASK        (0x6000)

#define _MM_FLUSH_ZERO_MASK   (0x8000)
#define _MM_FLUSH_ZERO_ON     (0x8000)
#define _MM_FLUSH_ZERO_OFF    (0x0000)

#define _MM_GET_EXCEPTION_MASK() (_mm_getcsr() & _MM_MASK_MASK)
#define _MM_GET_EXCEPTION_STATE() (_mm_getcsr() & _MM_EXCEPT_MASK)
#define _MM_GET_FLUSH_ZERO_MODE() (_mm_getcsr() & _MM_FLUSH_ZERO_MASK)
#define _MM_GET_ROUNDING_MODE() (_mm_getcsr() & _MM_ROUND_MASK)

#define _MM_SET_EXCEPTION_MASK(x) (_mm_setcsr((_mm_getcsr() & ~_MM_MASK_MASK) | (x)))
#define _MM_SET_EXCEPTION_STATE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_EXCEPT_MASK) | (x)))
#define _MM_SET_FLUSH_ZERO_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_FLUSH_ZERO_MASK) | (x)))
#define _MM_SET_ROUNDING_MODE(x) (_mm_setcsr((_mm_getcsr() & ~_MM_ROUND_MASK) | (x)))

#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movehl_ps(tmp3, tmp1); \
} while (0)

/* Aliases for compatibility. */
#define _m_pextrw _mm_extract_pi16
#define _m_pinsrw _mm_insert_pi16
#define _m_pmaxsw _mm_max_pi16
#define _m_pmaxub _mm_max_pu8
#define _m_pminsw _mm_min_pi16
#define _m_pminub _mm_min_pu8
#define _m_pmovmskb _mm_movemask_pi8
#define _m_pmulhuw _mm_mulhi_pu16
#define _m_pshufw _mm_shuffle_pi16
#define _m_maskmovq _mm_maskmove_si64
#define _m_pavgb _mm_avg_pu8
#define _m_pavgw _mm_avg_pu16
#define _m_psadbw _mm_sad_pu8
#define _m_ _mm_
#define _m_ _mm_

#undef __DEFAULT_FN_ATTRS

/* Ugly hack for backwards-compatibility (compatible with gcc) */
#if defined(__SSE2__) && !__has_feature(modules)
#include <emmintrin.h>
#endif

#endif /* __XMMINTRIN_H */
