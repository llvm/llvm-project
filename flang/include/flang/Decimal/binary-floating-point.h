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

#include "flang/Common/api-attrs.h"
#include "flang/Common/real.h"
#include "flang/Common/uint128.h"
#include <cinttypes>
#include <climits>
#include <cstring>
#include <type_traits>

namespace Fortran::decimal {

enum FortranRounding {
  RoundNearest, /* RN and RP */
  RoundUp, /* RU */
  RoundDown, /* RD */
  RoundToZero, /* RZ - no rounding */
  RoundCompatible, /* RC: like RN, but ties go away from 0 */
};

template <int BINARY_PRECISION> class BinaryFloatingPointNumber {
public:
  static constexpr common::RealCharacteristics realChars{BINARY_PRECISION};
  static constexpr int binaryPrecision{BINARY_PRECISION};
  static constexpr int bits{realChars.bits};
  static constexpr int isImplicitMSB{realChars.isImplicitMSB};
  static constexpr int significandBits{realChars.significandBits};
  static constexpr int exponentBits{realChars.exponentBits};
  static constexpr int exponentBias{realChars.exponentBias};
  static constexpr int maxExponent{realChars.maxExponent};
  static constexpr int decimalPrecision{realChars.decimalPrecision};
  static constexpr int decimalRange{realChars.decimalRange};
  static constexpr int maxDecimalConversionDigits{
      realChars.maxDecimalConversionDigits};

  using RawType = common::HostUnsignedIntType<bits>;
  static_assert(CHAR_BIT * sizeof(RawType) >= bits);
  RT_OFFLOAD_VAR_GROUP_BEGIN
  static constexpr RawType significandMask{(RawType{1} << significandBits) - 1};

  constexpr RT_API_ATTRS BinaryFloatingPointNumber() {} // zero
  RT_OFFLOAD_VAR_GROUP_END
  constexpr BinaryFloatingPointNumber(
      const BinaryFloatingPointNumber &that) = default;
  constexpr BinaryFloatingPointNumber(
      BinaryFloatingPointNumber &&that) = default;
  constexpr BinaryFloatingPointNumber &operator=(
      const BinaryFloatingPointNumber &that) = default;
  constexpr BinaryFloatingPointNumber &operator=(
      BinaryFloatingPointNumber &&that) = default;
  constexpr explicit RT_API_ATTRS BinaryFloatingPointNumber(RawType raw)
      : raw_{raw} {}

  RT_API_ATTRS RawType raw() const { return raw_; }

  template <typename A>
  explicit constexpr RT_API_ATTRS BinaryFloatingPointNumber(A x) {
    static_assert(sizeof raw_ <= sizeof x);
    std::memcpy(reinterpret_cast<void *>(&raw_),
        reinterpret_cast<const void *>(&x), sizeof raw_);
  }

  constexpr RT_API_ATTRS int BiasedExponent() const {
    return static_cast<int>(
        (raw_ >> significandBits) & ((1 << exponentBits) - 1));
  }
  constexpr RT_API_ATTRS int UnbiasedExponent() const {
    int biased{BiasedExponent()};
    return biased - exponentBias + (biased == 0);
  }
  constexpr RT_API_ATTRS RawType Significand() const {
    return raw_ & significandMask;
  }
  constexpr RT_API_ATTRS RawType Fraction() const {
    RawType sig{Significand()};
    if (isImplicitMSB && BiasedExponent() > 0) {
      sig |= RawType{1} << significandBits;
    }
    return sig;
  }

  constexpr RT_API_ATTRS bool IsZero() const {
    return (raw_ & ((RawType{1} << (bits - 1)) - 1)) == 0;
  }
  constexpr RT_API_ATTRS bool IsNaN() const {
    auto expo{BiasedExponent()};
    auto sig{Significand()};
    if constexpr (bits == 80) { // x87
      if (expo == maxExponent) {
        return sig != (significandMask >> 1) + 1;
      } else {
        return expo != 0 && !(sig & (RawType{1} << (significandBits - 1)));
        ;
      }
    } else {
      return expo == maxExponent && sig != 0;
    }
  }
  constexpr RT_API_ATTRS bool IsInfinite() const {
    if constexpr (bits == 80) { // x87
      return BiasedExponent() == maxExponent &&
          Significand() == ((significandMask >> 1) + 1);
    } else {
      return BiasedExponent() == maxExponent && Significand() == 0;
    }
  }
  constexpr RT_API_ATTRS bool IsMaximalFiniteMagnitude() const {
    return BiasedExponent() == maxExponent - 1 &&
        Significand() == significandMask;
  }
  constexpr RT_API_ATTRS bool IsNegative() const {
    return ((raw_ >> (bits - 1)) & 1) != 0;
  }

  constexpr RT_API_ATTRS void Negate() { raw_ ^= RawType{1} << (bits - 1); }

  // For calculating the nearest neighbors of a floating-point value
  constexpr RT_API_ATTRS void Previous() {
    RemoveExplicitMSB();
    --raw_;
    InsertExplicitMSB();
  }
  constexpr RT_API_ATTRS void Next() {
    RemoveExplicitMSB();
    ++raw_;
    InsertExplicitMSB();
  }

  static constexpr RT_API_ATTRS BinaryFloatingPointNumber Infinity(
      bool isNegative) {
    RawType result{RawType{maxExponent} << significandBits};
    if (isNegative) {
      result |= RawType{1} << (bits - 1);
    }
    return BinaryFloatingPointNumber{result};
  }

  // Returns true when the result is exact
  constexpr RT_API_ATTRS bool RoundToBits(
      int keepBits, enum FortranRounding mode) {
    if (IsNaN() || IsInfinite() || keepBits >= binaryPrecision) {
      return true;
    }
    int lostBits{keepBits < binaryPrecision ? binaryPrecision - keepBits : 0};
    RawType lostMask{static_cast<RawType>((RawType{1} << lostBits) - 1)};
    if (RawType lost{static_cast<RawType>(raw_ & lostMask)}; lost != 0) {
      bool increase{false};
      switch (mode) {
      case RoundNearest:
        if (lost >> (lostBits - 1) != 0) { // >= tie
          if ((lost & (lostMask >> 1)) != 0) {
            increase = true; // > tie
          } else {
            increase = ((raw_ >> lostBits) & 1) != 0; // tie to even
          }
        }
        break;
      case RoundUp:
        increase = !IsNegative();
        break;
      case RoundDown:
        increase = IsNegative();
        break;
      case RoundToZero:
        break;
      case RoundCompatible:
        increase = lost >> (lostBits - 1) != 0; // >= tie
        break;
      }
      if (increase) {
        raw_ |= lostMask;
        Next();
      }
      return false; // inexact
    } else {
      return true; // exact
    }
  }

private:
  constexpr RT_API_ATTRS void RemoveExplicitMSB() {
    if constexpr (!isImplicitMSB) {
      raw_ = (raw_ & (significandMask >> 1)) | ((raw_ & ~significandMask) >> 1);
    }
  }
  constexpr RT_API_ATTRS void InsertExplicitMSB() {
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
