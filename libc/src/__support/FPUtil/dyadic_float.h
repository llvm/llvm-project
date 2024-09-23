//===-- A class to store high precision floating point numbers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H

#include "FEnvImpl.h"
#include "FPBits.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "multiply_add.h"
#include "rounding_mode.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/big_int.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/macros/properties/types.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace fputil {

// A generic class to perform computations of high precision floating points.
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
  using MantissaType = LIBC_NAMESPACE::UInt<Bits>;

  Sign sign = Sign::POS;
  int exponent = 0;
  MantissaType mantissa = MantissaType(0);

  LIBC_INLINE constexpr DyadicFloat() = default;

  template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
  LIBC_INLINE constexpr DyadicFloat(T x) {
    static_assert(FPBits<T>::FRACTION_LEN < Bits);
    FPBits<T> x_bits(x);
    sign = x_bits.sign();
    exponent = x_bits.get_explicit_exponent() - FPBits<T>::FRACTION_LEN;
    mantissa = MantissaType(x_bits.get_explicit_mantissa());
    normalize();
  }

  LIBC_INLINE constexpr DyadicFloat(Sign s, int e, MantissaType m)
      : sign(s), exponent(e), mantissa(m) {
    normalize();
  }

  // Normalizing the mantissa, bringing the leading 1 bit to the most
  // significant bit.
  LIBC_INLINE constexpr DyadicFloat &normalize() {
    if (!mantissa.is_zero()) {
      int shift_length = cpp::countl_zero(mantissa);
      exponent -= shift_length;
      mantissa <<= static_cast<size_t>(shift_length);
    }
    return *this;
  }

  // Used for aligning exponents.  Output might not be normalized.
  LIBC_INLINE constexpr DyadicFloat &shift_left(unsigned shift_length) {
    if (shift_length < Bits) {
      exponent -= static_cast<int>(shift_length);
      mantissa <<= shift_length;
    } else {
      exponent = 0;
      mantissa = MantissaType(0);
    }
    return *this;
  }

  // Used for aligning exponents.  Output might not be normalized.
  LIBC_INLINE constexpr DyadicFloat &shift_right(unsigned shift_length) {
    if (shift_length < Bits) {
      exponent += static_cast<int>(shift_length);
      mantissa >>= shift_length;
    } else {
      exponent = 0;
      mantissa = MantissaType(0);
    }
    return *this;
  }

  // Assume that it is already normalized.  Output the unbiased exponent.
  LIBC_INLINE constexpr int get_unbiased_exponent() const {
    return exponent + (Bits - 1);
  }

#ifdef LIBC_TYPES_HAS_FLOAT16
  template <typename T, bool ShouldSignalExceptions>
  LIBC_INLINE constexpr cpp::enable_if_t<
      cpp::is_floating_point_v<T> && (FPBits<T>::FRACTION_LEN < Bits), T>
  generic_as() const {
    using FPBits = FPBits<float16>;
    using StorageType = typename FPBits::StorageType;

    constexpr int EXTRA_FRACTION_LEN = Bits - 1 - FPBits::FRACTION_LEN;

    if (mantissa == 0)
      return FPBits::zero(sign).get_val();

    int unbiased_exp = get_unbiased_exponent();

    if (unbiased_exp + FPBits::EXP_BIAS >= FPBits::MAX_BIASED_EXPONENT) {
      if constexpr (ShouldSignalExceptions) {
        set_errno_if_required(ERANGE);
        raise_except_if_required(FE_OVERFLOW | FE_INEXACT);
      }

      switch (quick_get_round()) {
      case FE_TONEAREST:
        return FPBits::inf(sign).get_val();
      case FE_TOWARDZERO:
        return FPBits::max_normal(sign).get_val();
      case FE_DOWNWARD:
        if (sign.is_pos())
          return FPBits::max_normal(Sign::POS).get_val();
        return FPBits::inf(Sign::NEG).get_val();
      case FE_UPWARD:
        if (sign.is_neg())
          return FPBits::max_normal(Sign::NEG).get_val();
        return FPBits::inf(Sign::POS).get_val();
      default:
        __builtin_unreachable();
      }
    }

    StorageType out_biased_exp = 0;
    StorageType out_mantissa = 0;
    bool round = false;
    bool sticky = false;
    bool underflow = false;

    if (unbiased_exp < -FPBits::EXP_BIAS - FPBits::FRACTION_LEN) {
      sticky = true;
      underflow = true;
    } else if (unbiased_exp == -FPBits::EXP_BIAS - FPBits::FRACTION_LEN) {
      round = true;
      MantissaType sticky_mask = (MantissaType(1) << (Bits - 1)) - 1;
      sticky = (mantissa & sticky_mask) != 0;
    } else {
      int extra_fraction_len = EXTRA_FRACTION_LEN;

      if (unbiased_exp < 1 - FPBits::EXP_BIAS) {
        underflow = true;
        extra_fraction_len += 1 - FPBits::EXP_BIAS - unbiased_exp;
      } else {
        out_biased_exp =
            static_cast<StorageType>(unbiased_exp + FPBits::EXP_BIAS);
      }

      MantissaType round_mask = MantissaType(1) << (extra_fraction_len - 1);
      round = (mantissa & round_mask) != 0;
      MantissaType sticky_mask = round_mask - 1;
      sticky = (mantissa & sticky_mask) != 0;

      out_mantissa = static_cast<StorageType>(mantissa >> extra_fraction_len);
    }

    bool lsb = (out_mantissa & 1) != 0;

    StorageType result =
        FPBits::create_value(sign, out_biased_exp, out_mantissa).uintval();

    switch (quick_get_round()) {
    case FE_TONEAREST:
      if (round && (lsb || sticky))
        ++result;
      break;
    case FE_DOWNWARD:
      if (sign.is_neg() && (round || sticky))
        ++result;
      break;
    case FE_UPWARD:
      if (sign.is_pos() && (round || sticky))
        ++result;
      break;
    default:
      break;
    }

    if (ShouldSignalExceptions && (round || sticky)) {
      int excepts = FE_INEXACT;
      if (FPBits(result).is_inf()) {
        set_errno_if_required(ERANGE);
        excepts |= FE_OVERFLOW;
      } else if (underflow) {
        set_errno_if_required(ERANGE);
        excepts |= FE_UNDERFLOW;
      }
      raise_except_if_required(excepts);
    }

    return FPBits(result).get_val();
  }
#endif // LIBC_TYPES_HAS_FLOAT16

  template <typename T, bool ShouldSignalExceptions,
            typename = cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                            (FPBits<T>::FRACTION_LEN < Bits),
                                        void>>
  LIBC_INLINE constexpr T fast_as() const {
    if (LIBC_UNLIKELY(mantissa.is_zero()))
      return FPBits<T>::zero(sign).get_val();

    // Assume that it is normalized, and output is also normal.
    constexpr uint32_t PRECISION = FPBits<T>::FRACTION_LEN + 1;
    using output_bits_t = typename FPBits<T>::StorageType;
    constexpr output_bits_t IMPLICIT_MASK =
        FPBits<T>::SIG_MASK - FPBits<T>::FRACTION_MASK;

    int exp_hi = exponent + static_cast<int>((Bits - 1) + FPBits<T>::EXP_BIAS);

    if (LIBC_UNLIKELY(exp_hi > 2 * FPBits<T>::EXP_BIAS)) {
      // Results overflow.
      T d_hi =
          FPBits<T>::create_value(sign, 2 * FPBits<T>::EXP_BIAS, IMPLICIT_MASK)
              .get_val();
      // volatile prevents constant propagation that would result in infinity
      // always being returned no matter the current rounding mode.
      volatile T two = static_cast<T>(2.0);
      T r = two * d_hi;

      // TODO: Whether rounding down the absolute value to max_normal should
      // also raise FE_OVERFLOW and set ERANGE is debatable.
      if (ShouldSignalExceptions && FPBits<T>(r).is_inf())
        set_errno_if_required(ERANGE);

      return r;
    }

    bool denorm = false;
    uint32_t shift = Bits - PRECISION;
    if (LIBC_UNLIKELY(exp_hi <= 0)) {
      // Output is denormal.
      denorm = true;
      shift = (Bits - PRECISION) + static_cast<uint32_t>(1 - exp_hi);

      exp_hi = FPBits<T>::EXP_BIAS;
    }

    int exp_lo = exp_hi - static_cast<int>(PRECISION) - 1;

    MantissaType m_hi =
        shift >= MantissaType::BITS ? MantissaType(0) : mantissa >> shift;

    T d_hi = FPBits<T>::create_value(
                 sign, static_cast<output_bits_t>(exp_hi),
                 (static_cast<output_bits_t>(m_hi) & FPBits<T>::SIG_MASK) |
                     IMPLICIT_MASK)
                 .get_val();

    MantissaType round_mask =
        shift > MantissaType::BITS ? 0 : MantissaType(1) << (shift - 1);
    MantissaType sticky_mask = round_mask - MantissaType(1);

    bool round_bit = !(mantissa & round_mask).is_zero();
    bool sticky_bit = !(mantissa & sticky_mask).is_zero();
    int round_and_sticky = int(round_bit) * 2 + int(sticky_bit);

    T d_lo;

    if (LIBC_UNLIKELY(exp_lo <= 0)) {
      // d_lo is denormal, but the output is normal.
      int scale_up_exponent = 1 - exp_lo;
      T scale_up_factor =
          FPBits<T>::create_value(Sign::POS,
                                  static_cast<output_bits_t>(
                                      FPBits<T>::EXP_BIAS + scale_up_exponent),
                                  IMPLICIT_MASK)
              .get_val();
      T scale_down_factor =
          FPBits<T>::create_value(Sign::POS,
                                  static_cast<output_bits_t>(
                                      FPBits<T>::EXP_BIAS - scale_up_exponent),
                                  IMPLICIT_MASK)
              .get_val();

      d_lo = FPBits<T>::create_value(
                 sign, static_cast<output_bits_t>(exp_lo + scale_up_exponent),
                 IMPLICIT_MASK)
                 .get_val();

      return multiply_add(d_lo, T(round_and_sticky), d_hi * scale_up_factor) *
             scale_down_factor;
    }

    d_lo = FPBits<T>::create_value(sign, static_cast<output_bits_t>(exp_lo),
                                   IMPLICIT_MASK)
               .get_val();

    // Still correct without FMA instructions if `d_lo` is not underflow.
    T r = multiply_add(d_lo, T(round_and_sticky), d_hi);

    if (LIBC_UNLIKELY(denorm)) {
      // Exponent before rounding is in denormal range, simply clear the
      // exponent field.
      output_bits_t clear_exp = static_cast<output_bits_t>(
          output_bits_t(exp_hi) << FPBits<T>::SIG_LEN);
      output_bits_t r_bits = FPBits<T>(r).uintval() - clear_exp;

      if (!(r_bits & FPBits<T>::EXP_MASK)) {
        // Output is denormal after rounding, clear the implicit bit for 80-bit
        // long double.
        r_bits -= IMPLICIT_MASK;

        // TODO: IEEE Std 754-2019 lets implementers choose whether to check for
        // "tininess" before or after rounding for base-2 formats, as long as
        // the same choice is made for all operations. Our choice to check after
        // rounding might not be the same as the hardware's.
        if (ShouldSignalExceptions && round_and_sticky) {
          set_errno_if_required(ERANGE);
          raise_except_if_required(FE_UNDERFLOW);
        }
      }

      return FPBits<T>(r_bits).get_val();
    }

    return r;
  }

  // Assume that it is already normalized.
  // Output is rounded correctly with respect to the current rounding mode.
  template <typename T, bool ShouldSignalExceptions,
            typename = cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                            (FPBits<T>::FRACTION_LEN < Bits),
                                        void>>
  LIBC_INLINE constexpr T as() const {
#if defined(LIBC_TYPES_HAS_FLOAT16) && !defined(__LIBC_USE_FLOAT16_CONVERSION)
    if constexpr (cpp::is_same_v<T, float16>)
      return generic_as<T, ShouldSignalExceptions>();
#endif
    return fast_as<T, ShouldSignalExceptions>();
  }

  template <typename T,
            typename = cpp::enable_if_t<cpp::is_floating_point_v<T> &&
                                            (FPBits<T>::FRACTION_LEN < Bits),
                                        void>>
  LIBC_INLINE explicit constexpr operator T() const {
    return as<T, /*ShouldSignalExceptions=*/false>();
  }

  LIBC_INLINE explicit constexpr operator MantissaType() const {
    if (mantissa.is_zero())
      return 0;

    MantissaType new_mant = mantissa;
    if (exponent > 0) {
      new_mant <<= exponent;
    } else {
      new_mant >>= (-exponent);
    }

    if (sign.is_neg()) {
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
LIBC_INLINE constexpr DyadicFloat<Bits> quick_add(DyadicFloat<Bits> a,
                                                  DyadicFloat<Bits> b) {
  if (LIBC_UNLIKELY(a.mantissa.is_zero()))
    return b;
  if (LIBC_UNLIKELY(b.mantissa.is_zero()))
    return a;

  // Align exponents
  if (a.exponent > b.exponent)
    b.shift_right(static_cast<unsigned>(a.exponent - b.exponent));
  else if (b.exponent > a.exponent)
    a.shift_right(static_cast<unsigned>(b.exponent - a.exponent));

  DyadicFloat<Bits> result;

  if (a.sign == b.sign) {
    // Addition
    result.sign = a.sign;
    result.exponent = a.exponent;
    result.mantissa = a.mantissa;
    if (result.mantissa.add_overflow(b.mantissa)) {
      // Mantissa addition overflow.
      result.shift_right(1);
      result.mantissa.val[DyadicFloat<Bits>::MantissaType::WORD_COUNT - 1] |=
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
//   2 * errors of quick_mul_hi = 2 * (UInt<Bits>::WORD_COUNT - 1) in ULPs.
// Assume inputs are normalized (by constructors or other functions) so that we
// don't need to normalize the inputs again in this function.  If the inputs are
// not normalized, the results might lose precision significantly.
template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits> quick_mul(const DyadicFloat<Bits> &a,
                                                  const DyadicFloat<Bits> &b) {
  DyadicFloat<Bits> result;
  result.sign = (a.sign != b.sign) ? Sign::NEG : Sign::POS;
  result.exponent = a.exponent + b.exponent + static_cast<int>(Bits);

  if (!(a.mantissa.is_zero() || b.mantissa.is_zero())) {
    result.mantissa = a.mantissa.quick_mul_hi(b.mantissa);
    // Check the leading bit directly, should be faster than using clz in
    // normalize().
    if (result.mantissa.val[DyadicFloat<Bits>::MantissaType::WORD_COUNT - 1] >>
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
LIBC_INLINE constexpr DyadicFloat<Bits>
multiply_add(const DyadicFloat<Bits> &a, const DyadicFloat<Bits> &b,
             const DyadicFloat<Bits> &c) {
  return quick_add(c, quick_mul(a, b));
}

// Simple exponentiation implementation for printf. Only handles positive
// exponents, since division isn't implemented.
template <size_t Bits>
LIBC_INLINE constexpr DyadicFloat<Bits> pow_n(const DyadicFloat<Bits> &a,
                                              uint32_t power) {
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
LIBC_INLINE constexpr DyadicFloat<Bits> mul_pow_2(const DyadicFloat<Bits> &a,
                                                  int32_t pow_2) {
  DyadicFloat<Bits> result = a;
  result.exponent += pow_2;
  return result;
}

} // namespace fputil
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_DYADIC_FLOAT_H
