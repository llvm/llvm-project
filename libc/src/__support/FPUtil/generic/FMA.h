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
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/UInt128.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

namespace LIBC_NAMESPACE {
namespace fputil {
namespace generic {

template <typename T> LIBC_INLINE T fma(T x, T y, T z);

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
            ? ((double(bit_sum) - double(bit_prod)) - double(bitz))
            : ((double(bit_sum) - double(bitz)) - double(bit_prod)));

    // Update sticky bits if t != 0.0 and the least (52 - 23 - 1 = 28) bits are
    // zero.
    if (!t.is_zero() && ((bit_sum.get_mantissa() & 0xfff'ffffULL) == 0)) {
      if (bit_sum.sign() != t.sign()) {
        bit_sum.set_mantissa(bit_sum.get_mantissa() + 1);
      } else if (bit_sum.get_mantissa()) {
        bit_sum.set_mantissa(bit_sum.get_mantissa() - 1);
      }
    }
  }

  return static_cast<float>(static_cast<double>(bit_sum));
}

namespace internal {

// Extract the sticky bits and shift the `mantissa` to the right by
// `shift_length`.
LIBC_INLINE bool shift_mantissa(int shift_length, UInt128 &mant) {
  if (shift_length >= 128) {
    mant = 0;
    return true; // prod_mant is non-zero.
  }
  UInt128 mask = (UInt128(1) << shift_length) - 1;
  bool sticky_bits = (mant & mask) != 0;
  mant >>= shift_length;
  return sticky_bits;
}

} // namespace internal

template <> LIBC_INLINE double fma<double>(double x, double y, double z) {
  using FPBits = fputil::FPBits<double>;

  if (LIBC_UNLIKELY(x == 0 || y == 0 || z == 0)) {
    return x * y + z;
  }

  int x_exp = 0;
  int y_exp = 0;
  int z_exp = 0;

  // Normalize denormal inputs.
  if (LIBC_UNLIKELY(FPBits(x).get_biased_exponent() == 0)) {
    x_exp -= 52;
    x *= 0x1.0p+52;
  }
  if (LIBC_UNLIKELY(FPBits(y).get_biased_exponent() == 0)) {
    y_exp -= 52;
    y *= 0x1.0p+52;
  }
  if (LIBC_UNLIKELY(FPBits(z).get_biased_exponent() == 0)) {
    z_exp -= 52;
    z *= 0x1.0p+52;
  }

  FPBits x_bits(x), y_bits(y), z_bits(z);
  const Sign z_sign = z_bits.sign();
  Sign prod_sign = (x_bits.sign() == y_bits.sign()) ? Sign::POS : Sign::NEG;
  x_exp += x_bits.get_biased_exponent();
  y_exp += y_bits.get_biased_exponent();
  z_exp += z_bits.get_biased_exponent();

  if (LIBC_UNLIKELY(x_exp == FPBits::MAX_BIASED_EXPONENT ||
                    y_exp == FPBits::MAX_BIASED_EXPONENT ||
                    z_exp == FPBits::MAX_BIASED_EXPONENT))
    return x * y + z;

  // Extract mantissa and append hidden leading bits.
  UInt128 x_mant = x_bits.get_mantissa() | FPBits::MIN_NORMAL;
  UInt128 y_mant = y_bits.get_mantissa() | FPBits::MIN_NORMAL;
  UInt128 z_mant = z_bits.get_mantissa() | FPBits::MIN_NORMAL;

  // If the exponent of the product x*y > the exponent of z, then no extra
  // precision beside the entire product x*y is needed.  On the other hand, when
  // the exponent of z >= the exponent of the product x*y, the worst-case that
  // we need extra precision is when there is cancellation and the most
  // significant bit of the product is aligned exactly with the second most
  // significant bit of z:
  //      z :    10aa...a
  // - prod :     1bb...bb....b
  // In that case, in order to store the exact result, we need at least
  //   (Length of prod) - (MantissaLength of z) = 2*(52 + 1) - 52 = 54.
  // Overall, before aligning the mantissas and exponents, we can simply left-
  // shift the mantissa of z by at least 54, and left-shift the product of x*y
  // by (that amount - 52).  After that, it is enough to align the least
  // significant bit, given that we keep track of the round and sticky bits
  // after the least significant bit.
  // We pick shifting z_mant by 64 bits so that technically we can simply use
  // the original mantissa as high part when constructing 128-bit z_mant. So the
  // mantissa of prod will be left-shifted by 64 - 54 = 10 initially.

  UInt128 prod_mant = x_mant * y_mant << 10;
  int prod_lsb_exp =
      x_exp + y_exp - (FPBits::EXP_BIAS + 2 * FPBits::FRACTION_LEN + 10);

  z_mant <<= 64;
  int z_lsb_exp = z_exp - (FPBits::FRACTION_LEN + 64);
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

  uint64_t result = 0;
  int r_exp = 0; // Unbiased exponent of the result

  // Normalize the result.
  if (prod_mant != 0) {
    uint64_t prod_hi = static_cast<uint64_t>(prod_mant >> 64);
    int lead_zeros =
        prod_hi ? cpp::countl_zero(prod_hi)
                : 64 + cpp::countl_zero(static_cast<uint64_t>(prod_mant));
    // Move the leading 1 to the most significant bit.
    prod_mant <<= lead_zeros;
    // The lower 64 bits are always sticky bits after moving the leading 1 to
    // the most significant bit.
    sticky_bits |= (static_cast<uint64_t>(prod_mant) != 0);
    result = static_cast<uint64_t>(prod_mant >> 64);
    // Change prod_lsb_exp the be the exponent of the least significant bit of
    // the result.
    prod_lsb_exp += 64 - lead_zeros;
    r_exp = prod_lsb_exp + 63;

    if (r_exp > 0) {
      // The result is normal.  We will shift the mantissa to the right by
      // 63 - 52 = 11 bits (from the locations of the most significant bit).
      // Then the rounding bit will correspond the 11th bit, and the lowest
      // 10 bits are merged into sticky bits.
      round_bit = (result & 0x0400ULL) != 0;
      sticky_bits |= (result & 0x03ffULL) != 0;
      result >>= 11;
    } else {
      if (r_exp < -52) {
        // The result is smaller than 1/2 of the smallest denormal number.
        sticky_bits = true; // since the result is non-zero.
        result = 0;
      } else {
        // The result is denormal.
        uint64_t mask = 1ULL << (11 - r_exp);
        round_bit = (result & mask) != 0;
        sticky_bits |= (result & (mask - 1)) != 0;
        if (r_exp > -52)
          result >>= 12 - r_exp;
        else
          result = 0;
      }

      r_exp = 0;
    }
  } else {
    // Return +0.0 when there is exact cancellation, i.e., x*y == -z exactly.
    prod_sign = Sign::POS;
  }

  // Finalize the result.
  int round_mode = fputil::quick_get_round();
  if (LIBC_UNLIKELY(r_exp >= FPBits::MAX_BIASED_EXPONENT)) {
    if ((round_mode == FE_TOWARDZERO) ||
        (round_mode == FE_UPWARD && prod_sign.is_neg()) ||
        (round_mode == FE_DOWNWARD && prod_sign.is_pos())) {
      result = FPBits::MAX_NORMAL;
      return prod_sign.is_neg() ? -cpp::bit_cast<double>(result)
                                : cpp::bit_cast<double>(result);
    }
    return static_cast<double>(FPBits::inf(prod_sign));
  }

  // Remove hidden bit and append the exponent field and sign bit.
  result = (result & FPBits::FRACTION_MASK) |
           (static_cast<uint64_t>(r_exp) << FPBits::FRACTION_LEN);
  if (prod_sign.is_neg()) {
    result |= FPBits::SIGN_MASK;
  }

  // Rounding.
  if (round_mode == FE_TONEAREST) {
    if (round_bit && (sticky_bits || ((result & 1) != 0)))
      ++result;
  } else if ((round_mode == FE_UPWARD && prod_sign.is_pos()) ||
             (round_mode == FE_DOWNWARD && prod_sign.is_neg())) {
    if (round_bit || sticky_bits)
      ++result;
  }

  return cpp::bit_cast<double>(result);
}

} // namespace generic
} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_FMA_H
