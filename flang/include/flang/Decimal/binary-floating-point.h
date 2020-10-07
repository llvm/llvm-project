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

template <int BINARY_PRECISION>
class BinaryFloatingPointNumber : public common::RealDetails<BINARY_PRECISION> {
public:
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

  constexpr BinaryFloatingPointNumber() {} // zero
  constexpr BinaryFloatingPointNumber(
      const BinaryFloatingPointNumber &that) = default;
  constexpr BinaryFloatingPointNumber(
      BinaryFloatingPointNumber &&that) = default;
  constexpr BinaryFloatingPointNumber &operator=(
      const BinaryFloatingPointNumber &that) = default;
  constexpr BinaryFloatingPointNumber &operator=(
      BinaryFloatingPointNumber &&that) = default;
  constexpr explicit BinaryFloatingPointNumber(RawType raw) : raw_{raw} {}

  RawType raw() const { return raw_; }

  template <typename A> explicit constexpr BinaryFloatingPointNumber(A x) {
    static_assert(sizeof raw_ <= sizeof x);
    std::memcpy(reinterpret_cast<void *>(&raw_),
        reinterpret_cast<const void *>(&x), sizeof raw_);
  }

  constexpr int BiasedExponent() const {
    return static_cast<int>(
        (raw_ >> significandBits) & ((1 << exponentBits) - 1));
  }
  constexpr int UnbiasedExponent() const {
    int biased{BiasedExponent()};
    return biased - exponentBias + (biased == 0);
  }
  constexpr RawType Significand() const { return raw_ & significandMask; }
  constexpr RawType Fraction() const {
    RawType sig{Significand()};
    if (isImplicitMSB && BiasedExponent() > 0) {
      sig |= RawType{1} << significandBits;
    }
    return sig;
  }

  constexpr bool IsZero() const {
    return (raw_ & ((RawType{1} << (bits - 1)) - 1)) == 0;
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
  constexpr bool IsNegative() const { return ((raw_ >> (bits - 1)) & 1) != 0; }

  constexpr void Negate() { raw_ ^= RawType{1} << (bits - 1); }

  // For calculating the nearest neighbors of a floating-point value
  constexpr void Previous() {
    RemoveExplicitMSB();
    --raw_;
    InsertExplicitMSB();
  }
  constexpr void Next() {
    RemoveExplicitMSB();
    ++raw_;
    InsertExplicitMSB();
  }

private:
  constexpr void RemoveExplicitMSB() {
    if constexpr (!isImplicitMSB) {
      raw_ = (raw_ & (significandMask >> 1)) | ((raw_ & ~significandMask) >> 1);
    }
  }
  constexpr void InsertExplicitMSB() {
    if constexpr (!isImplicitMSB) {
      constexpr RawType mask{significandMask >> 1};
      raw_ = (raw_ & mask) | ((raw_ & ~mask) << 1);
      if (BiasedExponent() > 0) {
        raw_ |= RawType{1} << (significandBits - 1);
      }
    }
  }

  RawType raw_{0};
};
} // namespace Fortran::decimal
#endif
