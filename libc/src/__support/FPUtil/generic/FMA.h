//===-- Common header for FMA implementations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_FMA_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_FMA_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/big_int.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include "hdr/fenv_macros.h"

namespace LIBC_NAMESPACE {
namespace fputil {
namespace generic {

template <typename OutType, typename InType>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<OutType> &&
                                 cpp::is_floating_point_v<InType> &&
                                 sizeof(OutType) <= sizeof(InType),
                             OutType>
fma(InType x, InType y, InType z);

// TODO(lntue): Implement fmaf that is correctly rounded to all rounding modes.
// The implementation below only is only correct for the default rounding mode,
// round-to-nearest tie-to-even.
template <> LIBC_INLINE float fma<float>(float x, float y, float z) {
  // Product is exact.
  double prod = static_cast<double>(x) * static_cast<double>(y);
  double z_d = static_cast<double>(z);
  double sum = prod + z_d;
  fputil::FPBits<double> bit_prod(prod), bitz(z_d), bit_sum(sum);

  if (!(bit_sum.is_inf_or_nan() || bit_sum.is_zero())) {
    // Since the sum is computed in double precision, rounding might happen
    // (for instance, when bitz.exponent > bit_prod.exponent + 5, or
    // bit_prod.exponent > bitz.exponent + 40).  In that case, when we round
    // the sum back to float, double rounding error might occur.
    // A concrete example of this phenomenon is as follows:
    //   x = y = 1 + 2^(-12), z = 2^(-53)
    // The exact value of x*y + z is 1 + 2^(-11) + 2^(-24) + 2^(-53)
    // So when rounding to float, fmaf(x, y, z) = 1 + 2^(-11) + 2^(-23)
    // On the other hand, with the default rounding mode,
    //   double(x*y + z) = 1 + 2^(-11) + 2^(-24)
    // and casting again to float gives us:
    //   float(double(x*y + z)) = 1 + 2^(-11).
    //
    // In order to correct this possible double rounding error, first we use
    // Dekker's 2Sum algorithm to find t such that sum - t = prod + z exactly,
    // assuming the (default) rounding mode is round-to-the-nearest,
    // tie-to-even.  Moreover, t satisfies the condition that t < eps(sum),
    // i.e., t.exponent < sum.exponent - 52. So if t is not 0, meaning rounding
    // occurs when computing the sum, we just need to use t to adjust (any) last
    // bit of sum, so that the sticky bits used when rounding sum to float are
    // correct (when it matters).
    fputil::FPBits<double> t(
        (bit_prod.get_biased_exponent() >= bitz.get_biased_exponent())
            ? ((bit_sum.get_val() - bit_prod.get_val()) - bitz.get_val())
            : ((bit_sum.get_val() - bitz.get_val()) - bit_prod.get_val()));

    // Update sticky bits if t != 0.0 and the least (52 - 23 - 1 = 28) bits are
    // zero.
    if (!t.is_zero() && ((bit_sum.get_mantissa() & 0xfff'ffffULL) == 0)) {
      if (bit_sum.sign() != t.sign())
        bit_sum.set_mantissa(bit_sum.get_mantissa() + 1);
      else if (bit_sum.get_mantissa())
        bit_sum.set_mantissa(bit_sum.get_mantissa() - 1);
    }
  }

  return static_cast<float>(bit_sum.get_val());
}

namespace internal {

// Extract the sticky bits and shift the `mantissa` to the right by
// `shift_length`.
template <typename T>
LIBC_INLINE cpp::enable_if_t<is_unsigned_integral_or_big_int_v<T>, bool>
shift_mantissa(int shift_length, T &mant) {
  if (shift_length >= cpp::numeric_limits<T>::digits) {
    mant = 0;
    return true; // prod_mant is non-zero.
  }
  T mask = (T(1) << shift_length) - 1;
  bool sticky_bits = (mant & mask) != 0;
  mant >>= shift_length;
  return sticky_bits;
}

} // namespace internal

template <typename OutType, typename InType>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<OutType> &&
                                 cpp::is_floating_point_v<InType> &&
                                 sizeof(OutType) <= sizeof(InType),
                             OutType>
fma(InType x, InType y, InType z) {
  using OutFPBits = fputil::FPBits<OutType>;
  using OutStorageType = typename OutFPBits::StorageType;
  using InFPBits = fputil::FPBits<InType>;
  using InStorageType = typename InFPBits::StorageType;

  constexpr int IN_EXPLICIT_MANT_LEN = InFPBits::FRACTION_LEN + 1;
  constexpr size_t PROD_LEN = 2 * IN_EXPLICIT_MANT_LEN;
  constexpr size_t TMP_RESULT_LEN = cpp::bit_ceil(PROD_LEN + 1);
  using TmpResultType = UInt<TMP_RESULT_LEN>;

  constexpr size_t EXTRA_FRACTION_LEN =
      TMP_RESULT_LEN - 1 - OutFPBits::FRACTION_LEN;
  constexpr TmpResultType EXTRA_FRACTION_STICKY_MASK =
      (TmpResultType(1) << (EXTRA_FRACTION_LEN - 1)) - 1;

  if (LIBC_UNLIKELY(x == 0 || y == 0 || z == 0))
    return static_cast<OutType>(x * y + z);

  int x_exp = 0;
  int y_exp = 0;
  int z_exp = 0;

  // Normalize denormal inputs.
  if (LIBC_UNLIKELY(InFPBits(x).is_subnormal())) {
    x_exp -= InFPBits::FRACTION_LEN;
    x *= InType(InStorageType(1) << InFPBits::FRACTION_LEN);
  }
  if (LIBC_UNLIKELY(InFPBits(y).is_subnormal())) {
    y_exp -= InFPBits::FRACTION_LEN;
    y *= InType(InStorageType(1) << InFPBits::FRACTION_LEN);
  }
  if (LIBC_UNLIKELY(InFPBits(z).is_subnormal())) {
    z_exp -= InFPBits::FRACTION_LEN;
    z *= InType(InStorageType(1) << InFPBits::FRACTION_LEN);
  }

  InFPBits x_bits(x), y_bits(y), z_bits(z);
  const Sign z_sign = z_bits.sign();
  Sign prod_sign = (x_bits.sign() == y_bits.sign()) ? Sign::POS : Sign::NEG;
  x_exp += x_bits.get_biased_exponent();
  y_exp += y_bits.get_biased_exponent();
  z_exp += z_bits.get_biased_exponent();

  if (LIBC_UNLIKELY(x_exp == InFPBits::MAX_BIASED_EXPONENT ||
                    y_exp == InFPBits::MAX_BIASED_EXPONENT ||
                    z_exp == InFPBits::MAX_BIASED_EXPONENT))
    return static_cast<OutType>(x * y + z);

  // Extract mantissa and append hidden leading bits.
  InStorageType x_mant = x_bits.get_explicit_mantissa();
  InStorageType y_mant = y_bits.get_explicit_mantissa();
  TmpResultType z_mant = z_bits.get_explicit_mantissa();

  // If the exponent of the product x*y > the exponent of z, then no extra
  // precision beside the entire product x*y is needed.  On the other hand, when
  // the exponent of z >= the exponent of the product x*y, the worst-case that
  // we need extra precision is when there is cancellation and the most
  // significant bit of the product is aligned exactly with the second most
  // significant bit of z:
  //      z :    10aa...a
  // - prod :     1bb...bb....b
  // In that case, in order to store the exact result, we need at least
  //     (Length of prod) - (Fraction length of z)
  //   = 2*(Length of input explicit mantissa) - (Fraction length of z) bits.
  // Overall, before aligning the mantissas and exponents, we can simply left-
  // shift the mantissa of z by that amount.  After that, it is enough to align
  // the least significant bit, given that we keep track of the round and sticky
  // bits after the least significant bit.

  TmpResultType prod_mant = TmpResultType(x_mant) * y_mant;
  int prod_lsb_exp =
      x_exp + y_exp - (InFPBits::EXP_BIAS + 2 * InFPBits::FRACTION_LEN);

  constexpr int RESULT_MIN_LEN = PROD_LEN - InFPBits::FRACTION_LEN;
  z_mant <<= RESULT_MIN_LEN;
  int z_lsb_exp = z_exp - (InFPBits::FRACTION_LEN + RESULT_MIN_LEN);
  bool round_bit = false;
  bool sticky_bits = false;
  bool z_shifted = false;

  // Align exponents.
  if (prod_lsb_exp < z_lsb_exp) {
    sticky_bits = internal::shift_mantissa(z_lsb_exp - prod_lsb_exp, prod_mant);
    prod_lsb_exp = z_lsb_exp;
  } else if (z_lsb_exp < prod_lsb_exp) {
    z_shifted = true;
    sticky_bits = internal::shift_mantissa(prod_lsb_exp - z_lsb_exp, z_mant);
  }

  // Perform the addition:
  //   (-1)^prod_sign * prod_mant + (-1)^z_sign * z_mant.
  // The final result will be stored in prod_sign and prod_mant.
  if (prod_sign == z_sign) {
    // Effectively an addition.
    prod_mant += z_mant;
  } else {
    // Subtraction cases.
    if (prod_mant >= z_mant) {
      if (z_shifted && sticky_bits) {
        // Add 1 more to the subtrahend so that the sticky bits remain
        // positive. This would simplify the rounding logic.
        ++z_mant;
      }
      prod_mant -= z_mant;
    } else {
      if (!z_shifted && sticky_bits) {
        // Add 1 more to the subtrahend so that the sticky bits remain
        // positive. This would simplify the rounding logic.
        ++prod_mant;
      }
      prod_mant = z_mant - prod_mant;
      prod_sign = z_sign;
    }
  }

  OutStorageType result = 0;
  int r_exp = 0; // Unbiased exponent of the result

  int round_mode = fputil::quick_get_round();

  // Normalize the result.
  if (prod_mant != 0) {
    int lead_zeros = cpp::countl_zero(prod_mant);
    // Move the leading 1 to the most significant bit.
    prod_mant <<= lead_zeros;
    prod_lsb_exp -= lead_zeros;
    r_exp = prod_lsb_exp + (cpp::numeric_limits<TmpResultType>::digits - 1) -
            InFPBits::EXP_BIAS + OutFPBits::EXP_BIAS;

    if (r_exp > 0) {
      // The result is normal.  We will shift the mantissa to the right by the
      // amount of extra bits compared to the length of the explicit mantissa in
      // the output type.  The rounding bit then becomes the highest bit that is
      // shifted out, and the following lower bits are merged into sticky bits.
      round_bit =
          (prod_mant & (TmpResultType(1) << (EXTRA_FRACTION_LEN - 1))) != 0;
      sticky_bits |= (prod_mant & EXTRA_FRACTION_STICKY_MASK) != 0;
      result = static_cast<OutStorageType>(prod_mant >> EXTRA_FRACTION_LEN);
    } else {
      if (r_exp < -OutFPBits::FRACTION_LEN) {
        // The result is smaller than 1/2 of the smallest denormal number.
        sticky_bits = true; // since the result is non-zero.
        result = 0;
      } else {
        // The result is denormal.
        TmpResultType mask = TmpResultType(1) << (EXTRA_FRACTION_LEN - r_exp);
        round_bit = (prod_mant & mask) != 0;
        sticky_bits |= (prod_mant & (mask - 1)) != 0;
        if (r_exp > -OutFPBits::FRACTION_LEN)
          result = static_cast<OutStorageType>(
              prod_mant >> (EXTRA_FRACTION_LEN + 1 - r_exp));
        else
          result = 0;
      }

      r_exp = 0;
    }
  } else {
    // When there is exact cancellation, i.e., x*y == -z exactly, return -0.0 if
    // rounding downward and +0.0 for other rounding modes.
    if (round_mode == FE_DOWNWARD)
      prod_sign = Sign::NEG;
    else
      prod_sign = Sign::POS;
  }

  // Finalize the result.
  if (LIBC_UNLIKELY(r_exp >= OutFPBits::MAX_BIASED_EXPONENT)) {
    if ((round_mode == FE_TOWARDZERO) ||
        (round_mode == FE_UPWARD && prod_sign.is_neg()) ||
        (round_mode == FE_DOWNWARD && prod_sign.is_pos())) {
      return OutFPBits::max_normal(prod_sign).get_val();
    }
    return OutFPBits::inf(prod_sign).get_val();
  }

  // Remove hidden bit and append the exponent field and sign bit.
  result = static_cast<OutStorageType>(
      (result & OutFPBits::FRACTION_MASK) |
      (static_cast<OutStorageType>(r_exp) << OutFPBits::FRACTION_LEN));
  if (prod_sign.is_neg())
    result |= OutFPBits::SIGN_MASK;

  // Rounding.
  if (round_mode == FE_TONEAREST) {
    if (round_bit && (sticky_bits || ((result & 1) != 0)))
      ++result;
  } else if ((round_mode == FE_UPWARD && prod_sign.is_pos()) ||
             (round_mode == FE_DOWNWARD && prod_sign.is_neg())) {
    if (round_bit || sticky_bits)
      ++result;
  }

  return cpp::bit_cast<OutType>(result);
}

} // namespace generic
} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_FMA_H
