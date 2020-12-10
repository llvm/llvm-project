//===-- Implementation of hypotf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_HYPOT_H
#define LLVM_LIBC_UTILS_FPUTIL_HYPOT_H

#include "BasicOperations.h"
#include "FPBits.h"
#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

namespace internal {

template <typename T> static inline T findLeadingOne(T mant, int &shift_length);

template <>
inline uint32_t findLeadingOne<uint32_t>(uint32_t mant, int &shift_length) {
  shift_length = 0;
  constexpr int nsteps = 5;
  constexpr uint32_t bounds[nsteps] = {1 << 16, 1 << 8, 1 << 4, 1 << 2, 1 << 1};
  constexpr int shifts[nsteps] = {16, 8, 4, 2, 1};
  for (int i = 0; i < nsteps; ++i) {
    if (mant >= bounds[i]) {
      shift_length += shifts[i];
      mant >>= shifts[i];
    }
  }
  return 1U << shift_length;
}

template <>
inline uint64_t findLeadingOne<uint64_t>(uint64_t mant, int &shift_length) {
  shift_length = 0;
  constexpr int nsteps = 6;
  constexpr uint64_t bounds[nsteps] = {1ULL << 32, 1ULL << 16, 1ULL << 8,
                                       1ULL << 4,  1ULL << 2,  1ULL << 1};
  constexpr int shifts[nsteps] = {32, 16, 8, 4, 2, 1};
  for (int i = 0; i < nsteps; ++i) {
    if (mant >= bounds[i]) {
      shift_length += shifts[i];
      mant >>= shifts[i];
    }
  }
  return 1ULL << shift_length;
}

} // namespace internal

template <typename T> struct DoubleLength;

template <> struct DoubleLength<uint16_t> { using Type = uint32_t; };

template <> struct DoubleLength<uint32_t> { using Type = uint64_t; };

template <> struct DoubleLength<uint64_t> { using Type = __uint128_t; };

// Correctly rounded IEEE 754 HYPOT(x, y) with round to nearest, ties to even.
//
// Algorithm:
//   -  Let a = max(|x|, |y|), b = min(|x|, |y|), then we have that:
//          a <= sqrt(a^2 + b^2) <= min(a + b, a*sqrt(2))
//   1. So if b < eps(a)/2, then HYPOT(x, y) = a.
//
//   -  Moreover, the exponent part of HYPOT(x, y) is either the same or 1 more
//      than the exponent part of a.
//
//   2. For the remaining cases, we will use the digit-by-digit (shift-and-add)
//      algorithm to compute SQRT(Z):
//
//   -  For Y = y0.y1...yn... = SQRT(Z),
//      let Y(n) = y0.y1...yn be the first n fractional digits of Y.
//
//   -  The nth scaled residual R(n) is defined to be:
//          R(n) = 2^n * (Z - Y(n)^2)
//
//   -  Since Y(n) = Y(n - 1) + yn * 2^(-n), the scaled residual
//      satisfies the following recurrence formula:
//          R(n) = 2*R(n - 1) - yn*(2*Y(n - 1) + 2^(-n)),
//      with the initial conditions:
//          Y(0) = y0, and R(0) = Z - y0.
//
//   -  So the nth fractional digit of Y = SQRT(Z) can be decided by:
//          yn = 1  if 2*R(n - 1) >= 2*Y(n - 1) + 2^(-n),
//               0  otherwise.
//
//   3. Precision analysis:
//
//   -  Notice that in the decision function:
//          2*R(n - 1) >= 2*Y(n - 1) + 2^(-n),
//      the right hand side only uses up to the 2^(-n)-bit, and both sides are
//      non-negative, so R(n - 1) can be truncated at the 2^(-(n + 1))-bit, so
//      that 2*R(n - 1) is corrected up to the 2^(-n)-bit.
//
//   -  Thus, in order to round SQRT(a^2 + b^2) correctly up to n-fractional
//      bits, we need to perform the summation (a^2 + b^2) correctly up to (2n +
//      2)-fractional bits, and the remaining bits are sticky bits (i.e. we only
//      care if they are 0 or > 0), and the comparisons, additions/subtractions
//      can be done in n-fractional bits precision.
//
//   -  For single precision (float), we can use uint64_t to store the sum a^2 +
//      b^2 exact up to (2n + 2)-fractional bits.
//
//   -  Then we can feed this sum into the digit-by-digit algorithm for SQRT(Z)
//      described above.
//
//
// Special cases:
//   - HYPOT(x, y) is +Inf if x or y is +Inf or -Inf; else
//   - HYPOT(x, y) is NaN if x or y is NaN.
//
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T hypot(T x, T y) {
  using FPBits_t = FPBits<T>;
  using UIntType = typename FPBits<T>::UIntType;
  using DUIntType = typename DoubleLength<UIntType>::Type;

  FPBits_t x_bits(x), y_bits(y);

  if (x_bits.isInf() || y_bits.isInf()) {
    return FPBits_t::inf();
  }
  if (x_bits.isNaN()) {
    return x;
  }
  if (y_bits.isNaN()) {
    return y;
  }

  uint16_t a_exp, b_exp, out_exp;
  UIntType a_mant, b_mant;
  DUIntType a_mant_sq, b_mant_sq;
  bool sticky_bits;

  if ((x_bits.exponent >= y_bits.exponent + MantissaWidth<T>::value + 2) ||
      (y == 0)) {
    return abs(x);
  } else if ((y_bits.exponent >=
              x_bits.exponent + MantissaWidth<T>::value + 2) ||
             (x == 0)) {
    y_bits.sign = 0;
    return abs(y);
  }

  if (x >= y) {
    a_exp = x_bits.exponent;
    a_mant = x_bits.mantissa;
    b_exp = y_bits.exponent;
    b_mant = y_bits.mantissa;
  } else {
    a_exp = y_bits.exponent;
    a_mant = y_bits.mantissa;
    b_exp = x_bits.exponent;
    b_mant = x_bits.mantissa;
  }

  out_exp = a_exp;

  // Add an extra bit to simplify the final rounding bit computation.
  constexpr UIntType one = UIntType(1) << (MantissaWidth<T>::value + 1);

  a_mant <<= 1;
  b_mant <<= 1;

  UIntType leading_one;
  int y_mant_width;
  if (a_exp != 0) {
    leading_one = one;
    a_mant |= one;
    y_mant_width = MantissaWidth<T>::value + 1;
  } else {
    leading_one = internal::findLeadingOne(a_mant, y_mant_width);
  }

  if (b_exp != 0) {
    b_mant |= one;
  }

  a_mant_sq = static_cast<DUIntType>(a_mant) * a_mant;
  b_mant_sq = static_cast<DUIntType>(b_mant) * b_mant;

  // At this point, a_exp >= b_exp > a_exp - 25, so in order to line up aSqMant
  // and bSqMant, we need to shift bSqMant to the right by (a_exp - b_exp) bits.
  // But before that, remember to store the losing bits to sticky.
  // The shift length is for a^2 and b^2, so it's double of the exponent
  // difference between a and b.
  uint16_t shift_length = 2 * (a_exp - b_exp);
  sticky_bits =
      ((b_mant_sq & ((DUIntType(1) << shift_length) - DUIntType(1))) !=
       DUIntType(0));
  b_mant_sq >>= shift_length;

  DUIntType sum = a_mant_sq + b_mant_sq;
  if (sum >= (DUIntType(1) << (2 * y_mant_width + 2))) {
    // a^2 + b^2 >= 4* leading_one^2, so we will need an extra bit to the left.
    if (leading_one == one) {
      // For normal result, we discard the last 2 bits of the sum and increase
      // the exponent.
      sticky_bits = sticky_bits || ((sum & 0x3U) != 0);
      sum >>= 2;
      ++out_exp;
      if (out_exp >= FPBits_t::maxExponent) {
        return FPBits_t::inf();
      }
    } else {
      // For denormal result, we simply move the leading bit of the result to
      // the left by 1.
      leading_one <<= 1;
      ++y_mant_width;
    }
  }

  UIntType Y = leading_one;
  UIntType R = static_cast<UIntType>(sum >> y_mant_width) - leading_one;
  UIntType tailBits = static_cast<UIntType>(sum) & (leading_one - 1);

  for (UIntType current_bit = leading_one >> 1; current_bit;
       current_bit >>= 1) {
    R = (R << 1) + ((tailBits & current_bit) ? 1 : 0);
    UIntType tmp = (Y << 1) + current_bit; // 2*y(n - 1) + 2^(-n)
    if (R >= tmp) {
      R -= tmp;
      Y += current_bit;
    }
  }

  bool round_bit = Y & UIntType(1);
  bool lsb = Y & UIntType(2);

  if (Y >= one) {
    Y -= one;

    if (out_exp == 0) {
      out_exp = 1;
    }
  }

  Y >>= 1;

  // Round to the nearest, tie to even.
  if (round_bit && (lsb || sticky_bits || (R != 0))) {
    ++Y;
  }

  if (Y >= (one >> 1)) {
    Y -= one >> 1;
    ++out_exp;
    if (out_exp >= FPBits_t::maxExponent) {
      return FPBits_t::inf();
    }
  }

  Y |= static_cast<UIntType>(out_exp) << MantissaWidth<T>::value;
  return *reinterpret_cast<T *>(&Y);
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_HYPOT_H
