//===-- Implementation of hypotf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/math/hypotf.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, hypotf, (float x, float y)) {
  using DoubleBits = fputil::FPBits<double>;
  using FPBits = fputil::FPBits<float>;

  FPBits x_bits(x), y_bits(y);

  uint16_t x_exp = x_bits.get_unbiased_exponent();
  uint16_t y_exp = y_bits.get_unbiased_exponent();
  uint16_t exp_diff = (x_exp > y_exp) ? (x_exp - y_exp) : (y_exp - x_exp);

  if (exp_diff >= fputil::MantissaWidth<float>::VALUE + 2) {
    return fputil::abs(x) + fputil::abs(y);
  }

  double xd = static_cast<double>(x);
  double yd = static_cast<double>(y);

  // These squares are exact.
  double x_sq = xd * xd;
  double y_sq = yd * yd;

  // Compute the sum of squares.
  double sum_sq = x_sq + y_sq;

  // Compute the rounding error with Fast2Sum algorithm:
  // x_sq + y_sq = sum_sq - err
  double err = (x_sq >= y_sq) ? (sum_sq - x_sq) - y_sq : (sum_sq - y_sq) - x_sq;

  // Take sqrt in double precision.
  DoubleBits result(fputil::sqrt(sum_sq));

  if (!DoubleBits(sum_sq).is_inf_or_nan()) {
    // Correct rounding.
    double r_sq = static_cast<double>(result) * static_cast<double>(result);
    double diff = sum_sq - r_sq;
    constexpr uint64_t mask = 0x0000'0000'3FFF'FFFFULL;
    uint64_t lrs = result.uintval() & mask;

    if (lrs == 0x0000'0000'1000'0000ULL && err < diff) {
      result.bits |= 1ULL;
    } else if (lrs == 0x0000'0000'3000'0000ULL && err > diff) {
      result.bits -= 1ULL;
    }
  } else {
    FPBits bits_x(x), bits_y(y);
    if (bits_x.is_inf_or_nan() || bits_y.is_inf_or_nan()) {
      if (bits_x.is_inf() || bits_y.is_inf())
        return static_cast<float>(FPBits::inf());
      if (bits_x.is_nan())
        return x;
      return y;
    }
  }

  return static_cast<float>(static_cast<double>(result));
}

} // namespace __llvm_libc
