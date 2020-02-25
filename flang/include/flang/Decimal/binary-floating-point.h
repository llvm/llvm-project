//===-- include/flang/Decimal/binary-floating-point.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DECIMAL_BINARY_FLOATING_POINT_H_
#define FORTRAN_DECIMAL_BINARY_FLOATING_POINT_H_

// Access and manipulate the fields of an IEEE-754 binary
// floating-point value via a generalized template.

#include "flang/Common/real.h"
#include "flang/Common/uint128.h"
#include <cinttypes>
#include <climits>
#include <cstring>
#include <type_traits>

namespace Fortran::decimal {

template<int BINARY_PRECISION>
struct BinaryFloatingPointNumber
  : public common::RealDetails<BINARY_PRECISION> {

  using Details = common::RealDetails<BINARY_PRECISION>;
  using Details::bits;
  using Details::decimalPrecision;
  using Details::decimalRange;
  using Details::exponentBias;
  using Details::exponentBits;
  using Details::isImplicitMSB;
  using Details::maxDecimalConversionDigits;
  using Details::maxExponent;
  using Details::significandBits;

  using RawType = common::HostUnsignedIntType<bits>;
  static_assert(CHAR_BIT * sizeof(RawType) >= bits);
  static constexpr RawType significandMask{(RawType{1} << significandBits) - 1};

  constexpr BinaryFloatingPointNumber() {}  // zero
  constexpr BinaryFloatingPointNumber(
      const BinaryFloatingPointNumber &that) = default;
  constexpr BinaryFloatingPointNumber(
      BinaryFloatingPointNumber &&that) = default;
  constexpr BinaryFloatingPointNumber &operator=(
      const BinaryFloatingPointNumber &that) = default;
  constexpr BinaryFloatingPointNumber &operator=(
      BinaryFloatingPointNumber &&that) = default;

  template<typename A> explicit constexpr BinaryFloatingPointNumber(A x) {
    static_assert(sizeof raw <= sizeof x);
    std::memcpy(reinterpret_cast<void *>(&raw),
        reinterpret_cast<const void *>(&x), sizeof raw);
  }

  constexpr int BiasedExponent() const {
    return static_cast<int>(
        (raw >> significandBits) & ((1 << exponentBits) - 1));
  }
  constexpr int UnbiasedExponent() const {
    int biased{BiasedExponent()};
    return biased - exponentBias + (biased == 0);
  }
  constexpr RawType Significand() const { return raw & significandMask; }
  constexpr RawType Fraction() const {
    RawType sig{Significand()};
    if (isImplicitMSB && BiasedExponent() > 0) {
      sig |= RawType{1} << significandBits;
    }
    return sig;
  }

  constexpr bool IsZero() const {
    return (raw & ((RawType{1} << (bits - 1)) - 1)) == 0;
  }
  constexpr bool IsNaN() const {
    return BiasedExponent() == maxExponent && Significand() != 0;
  }
  constexpr bool IsInfinite() const {
    return BiasedExponent() == maxExponent && Significand() == 0;
  }
  constexpr bool IsMaximalFiniteMagnitude() const {
    return BiasedExponent() == maxExponent - 1 &&
        Significand() == significandMask;
  }
  constexpr bool IsNegative() const { return ((raw >> (bits - 1)) & 1) != 0; }

  constexpr void Negate() { raw ^= RawType{1} << (bits - 1); }

  RawType raw{0};
};
}
#endif
