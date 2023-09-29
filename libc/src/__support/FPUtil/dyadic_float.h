//===-- A class to store high precision floating point numbers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H

#include "FPBits.h"
#include "FloatProperties.h"
#include "multiply_add.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/UInt.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include <stddef.h>

namespace LIBC_NAMESPACE::fputil {

// A generic class to perform comuptations of high precision floating points.
// We store the value in dyadic format, including 3 fields:
//   sign    : boolean value - false means positive, true means negative
//   exponent: the exponent value of the least significant bit of the mantissa.
//   mantissa: unsigned integer of length `Bits`.
// So the real value that is stored is:
//   real value = (-1)^sign * 2^exponent * (mantissa as unsigned integer)
// The stored data is normal if for non-zero mantissa, the leading bit is 1.
// The outputs of the constructors and most functions will be normalized.
// To simplify and improve the efficiency, many functions will assume that the
// inputs are normal.
template <size_t Bits> struct DyadicFloat {
  using MantissaType = LIBC_NAMESPACE::cpp::UInt<Bits>;

  bool sign = false;
  int exponent = 0;
  MantissaType mantissa = MantissaType(0);

  DyadicFloat() = default;

  template <typename T,
            cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                 (FloatProperties<T>::MANTISSA_WIDTH < Bits),
                             int> = 0>
  DyadicFloat(T x) {
    FPBits<T> x_bits(x);
    sign = x_bits.get_sign();
    exponent = x_bits.get_exponent() - FloatProperties<T>::MANTISSA_WIDTH;
    mantissa = MantissaType(x_bits.get_explicit_mantissa());
    normalize();
  }

  constexpr DyadicFloat(bool s, int e, MantissaType m)
      : sign(s), exponent(e), mantissa(m) {
    normalize();
  }

  // Normalizing the mantissa, bringing the leading 1 bit to the most
  // significant bit.
  constexpr DyadicFloat &normalize() {
    if (!mantissa.is_zero()) {
      int shift_length = static_cast<int>(mantissa.clz());
      exponent -= shift_length;
      mantissa.shift_left(static_cast<size_t>(shift_length));
    }
    return *this;
  }

  // Used for aligning exponents.  Output might not be normalized.
  DyadicFloat &shift_left(int shift_length) {
    exponent -= shift_length;
    mantissa <<= static_cast<size_t>(shift_length);
    return *this;
  }

  // Used for aligning exponents.  Output might not be normalized.
  DyadicFloat &shift_right(int shift_length) {
    exponent += shift_length;
    mantissa >>= static_cast<size_t>(shift_length);
    return *this;
  }

  // Assume that it is already normalized and output is not underflow.
  // Output is rounded correctly with respect to the current rounding mode.
  // TODO(lntue): Add support for underflow.
  // TODO(lntue): Test or add specialization for x86 long double.
  template <typename T, typename = cpp::enable_if_t<
                            cpp::is_floating_point_v<T> &&
                                (FloatProperties<T>::MANTISSA_WIDTH < Bits),
                            void>>
  explicit operator T() const {
    // TODO(lntue): Do we need to treat signed zeros properly?
    if (mantissa.is_zero())
      return 0.0;

    // Assume that it is normalized, and output is also normal.
    constexpr size_t PRECISION = FloatProperties<T>::MANTISSA_WIDTH + 1;
    using output_bits_t = typename FPBits<T>::UIntType;

    int exp_hi = exponent + static_cast<int>((Bits - 1) +
                                             FloatProperties<T>::EXPONENT_BIAS);

    bool denorm = false;
    uint32_t shift = Bits - PRECISION;
    if (LIBC_UNLIKELY(exp_hi <= 0)) {
      // Output is denormal.
      denorm = true;
      shift = (Bits - PRECISION) + static_cast<uint32_t>(1 - exp_hi);

      exp_hi = FloatProperties<T>::EXPONENT_BIAS;
    }

    int exp_lo = exp_hi - PRECISION - 1;

    MantissaType m_hi(mantissa >> shift);

    T d_hi = FPBits<T>::create_value(sign, exp_hi,
                                     output_bits_t(m_hi) &
                                         FloatProperties<T>::MANTISSA_MASK)
                 .get_val();

    const MantissaType round_mask = MantissaType(1) << (shift - 1);
    const MantissaType sticky_mask = round_mask - MantissaType(1);

    bool round_bit = !(mantissa & round_mask).is_zero();
    bool sticky_bit = !(mantissa & sticky_mask).is_zero();
    int round_and_sticky = int(round_bit) * 2 + int(sticky_bit);

    T d_lo;
    if (LIBC_UNLIKELY(exp_lo <= 0)) {
      // d_lo is denormal, but the output is normal.
      int scale_up_exponent = 2 * PRECISION;
      T scale_up_factor =
          FPBits<T>::create_value(
              sign, FloatProperties<T>::EXPONENT_BIAS + scale_up_exponent,
              output_bits_t(0))
              .get_val();
      T scale_down_factor =
          FPBits<T>::create_value(
              sign, FloatProperties<T>::EXPONENT_BIAS - scale_up_exponent,
              output_bits_t(0))
              .get_val();

      d_lo = FPBits<T>::create_value(sign, exp_lo + scale_up_exponent,
                                     output_bits_t(0))
                 .get_val();

      return multiply_add(d_lo, T(round_and_sticky), d_hi * scale_up_factor) *
             scale_down_factor;
    }

    d_lo = FPBits<T>::create_value(sign, exp_lo, output_bits_t(0)).get_val();

    // Still correct without FMA instructions if `d_lo` is not underflow.
    T r = multiply_add(d_lo, T(round_and_sticky), d_hi);

    if (LIBC_UNLIKELY(denorm)) {
      // Output is denormal, simply clear the exponent field.
      output_bits_t clear_exp = output_bits_t(exp_hi)
                                << FloatProperties<T>::MANTISSA_WIDTH;
      output_bits_t r_bits = FPBits<T>(r).uintval() - clear_exp;
      return FPBits<T>(r_bits).get_val();
    }

    return r;
  }

  explicit operator MantissaType() const {
    if (mantissa.is_zero())
      return 0;

    MantissaType new_mant = mantissa;
    if (exponent > 0) {
      new_mant <<= exponent;
    } else {
      new_mant >>= (-exponent);
    }

    if (sign) {
      new_mant = (~new_mant) + 1;
    }

    return new_mant;
  }
};

// Quick add - Add 2 dyadic floats with rounding toward 0 and then normalize the
// output:
//   - Align the exponents so that:
//     new a.exponent = new b.exponent = max(a.exponent, b.exponent)
//   - Add or subtract the mantissas depending on the signs.
//   - Normalize the result.
// The absolute errors compared to the mathematical sum is bounded by:
//   | quick_add(a, b) - (a + b) | < MSB(a + b) * 2^(-Bits + 2),
// i.e., errors are up to 2 ULPs.
// Assume inputs are normalized (by constructors or other functions) so that we
// don't need to normalize the inputs again in this function.  If the inputs are
// not normalized, the results might lose precision significantly.
template <size_t Bits>
constexpr DyadicFloat<Bits> quick_add(DyadicFloat<Bits> a,
                                      DyadicFloat<Bits> b) {
  if (LIBC_UNLIKELY(a.mantissa.is_zero()))
    return b;
  if (LIBC_UNLIKELY(b.mantissa.is_zero()))
    return a;

  // Align exponents
  if (a.exponent > b.exponent)
    b.shift_right(a.exponent - b.exponent);
  else if (b.exponent > a.exponent)
    a.shift_right(b.exponent - a.exponent);

  DyadicFloat<Bits> result;

  if (a.sign == b.sign) {
    // Addition
    result.sign = a.sign;
    result.exponent = a.exponent;
    result.mantissa = a.mantissa;
    if (result.mantissa.add(b.mantissa)) {
      // Mantissa addition overflow.
      result.shift_right(1);
      result.mantissa.val[DyadicFloat<Bits>::MantissaType::WORDCOUNT - 1] |=
          (uint64_t(1) << 63);
    }
    // Result is already normalized.
    return result;
  }

  // Subtraction
  if (a.mantissa >= b.mantissa) {
    result.sign = a.sign;
    result.exponent = a.exponent;
    result.mantissa = a.mantissa - b.mantissa;
  } else {
    result.sign = b.sign;
    result.exponent = b.exponent;
    result.mantissa = b.mantissa - a.mantissa;
  }

  return result.normalize();
}

// Quick Mul - Slightly less accurate but efficient multiplication of 2 dyadic
// floats with rounding toward 0 and then normalize the output:
//   result.exponent = a.exponent + b.exponent + Bits,
//   result.mantissa = quick_mul_hi(a.mantissa + b.mantissa)
//                   ~ (full product a.mantissa * b.mantissa) >> Bits.
// The errors compared to the mathematical product is bounded by:
//   2 * errors of quick_mul_hi = 2 * (UInt<Bits>::WORDCOUNT - 1) in ULPs.
// Assume inputs are normalized (by constructors or other functions) so that we
// don't need to normalize the inputs again in this function.  If the inputs are
// not normalized, the results might lose precision significantly.
template <size_t Bits>
constexpr DyadicFloat<Bits> quick_mul(DyadicFloat<Bits> a,
                                      DyadicFloat<Bits> b) {
  DyadicFloat<Bits> result;
  result.sign = (a.sign != b.sign);
  result.exponent = a.exponent + b.exponent + int(Bits);

  if (!(a.mantissa.is_zero() || b.mantissa.is_zero())) {
    result.mantissa = a.mantissa.quick_mul_hi(b.mantissa);
    // Check the leading bit directly, should be faster than using clz in
    // normalize().
    if (result.mantissa.val[DyadicFloat<Bits>::MantissaType::WORDCOUNT - 1] >>
            63 ==
        0)
      result.shift_left(1);
  } else {
    result.mantissa = (typename DyadicFloat<Bits>::MantissaType)(0);
  }
  return result;
}

// Simple polynomial approximation.
template <size_t Bits>
constexpr DyadicFloat<Bits> multiply_add(const DyadicFloat<Bits> &a,
                                         const DyadicFloat<Bits> &b,
                                         const DyadicFloat<Bits> &c) {
  return quick_add(c, quick_mul(a, b));
}

// Simple exponentiation implementation for printf. Only handles positive
// exponents, since division isn't implemented.
template <size_t Bits>
constexpr DyadicFloat<Bits> pow_n(DyadicFloat<Bits> a, uint32_t power) {
  DyadicFloat<Bits> result = 1.0;
  DyadicFloat<Bits> cur_power = a;

  while (power > 0) {
    if ((power % 2) > 0) {
      result = quick_mul(result, cur_power);
    }
    power = power >> 1;
    cur_power = quick_mul(cur_power, cur_power);
  }
  return result;
}

template <size_t Bits>
constexpr DyadicFloat<Bits> mul_pow_2(DyadicFloat<Bits> a, int32_t pow_2) {
  DyadicFloat<Bits> result = a;
  result.exponent += pow_2;
  return result;
}

} // namespace LIBC_NAMESPACE::fputil

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H
