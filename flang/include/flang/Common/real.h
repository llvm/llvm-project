//===-- include/flang/Common/real.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_REAL_H_
#define FORTRAN_COMMON_REAL_H_

// Characteristics of IEEE-754 & related binary floating-point numbers.
// The various representations are distinguished by their binary precisions
// (number of explicit significand bits and any implicit MSB in the fraction).

#include <cinttypes>

namespace Fortran::common {

// Total representation size in bits for each type
static constexpr int BitsForBinaryPrecision(int binaryPrecision) {
  switch (binaryPrecision) {
  case 8:
    return 16; // IEEE single (truncated): 1+8+7
  case 11:
    return 16; // IEEE half precision: 1+5+10
  case 24:
    return 32; // IEEE single precision: 1+8+23
  case 53:
    return 64; // IEEE double precision: 1+11+52
  case 64:
    return 80; // x87 extended precision: 1+15+64
  case 106:
    return 128; // "double-double": 2*(1+11+52)
  case 113:
    return 128; // IEEE quad precision: 1+15+112
  default:
    return -1;
  }
}

// Maximum number of significant decimal digits in the fraction of an
// exact conversion in each type; computed by converting the value
// with the minimum exponent (biased to 1) and all fractional bits set.
static constexpr int MaxDecimalConversionDigits(int binaryPrecision) {
  switch (binaryPrecision) {
  case 8:
    return 96;
  case 11:
    return 21;
  case 24:
    return 112;
  case 53:
    return 767;
  case 64:
    return 11514;
  case 106:
    return 2 * 767;
  case 113:
    return 11563;
  default:
    return -1;
  }
}

template <int BINARY_PRECISION> class RealDetails {
private:
  // Converts bit widths to whole decimal digits
  static constexpr int LogBaseTwoToLogBaseTen(int logb2) {
    constexpr std::int64_t LogBaseTenOfTwoTimesTenToThe12th{301029995664};
    constexpr std::int64_t TenToThe12th{1000000000000};
    std::int64_t logb10{
        (logb2 * LogBaseTenOfTwoTimesTenToThe12th) / TenToThe12th};
    return static_cast<int>(logb10);
  }

public:
  static constexpr int binaryPrecision{BINARY_PRECISION};
  static constexpr int bits{BitsForBinaryPrecision(binaryPrecision)};
  static constexpr bool isImplicitMSB{binaryPrecision != 64 /*x87*/};
  static constexpr int significandBits{binaryPrecision - isImplicitMSB};
  static constexpr int exponentBits{bits - significandBits - 1 /*sign*/};
  static constexpr int maxExponent{(1 << exponentBits) - 1};
  static constexpr int exponentBias{maxExponent / 2};

  static constexpr int decimalPrecision{
      LogBaseTwoToLogBaseTen(binaryPrecision - 1)};
  static constexpr int decimalRange{LogBaseTwoToLogBaseTen(exponentBias - 1)};

  // Number of significant decimal digits in the fraction of the
  // exact conversion of the least nonzero subnormal.
  static constexpr int maxDecimalConversionDigits{
      MaxDecimalConversionDigits(binaryPrecision)};

  static_assert(binaryPrecision > 0);
  static_assert(exponentBits > 1);
  static_assert(exponentBits <= 15);
};

} // namespace Fortran::common
#endif // FORTRAN_COMMON_REAL_H_
