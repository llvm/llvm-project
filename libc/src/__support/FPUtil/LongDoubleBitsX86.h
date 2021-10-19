//===-- Bit representation of x86 long double numbers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_LONG_DOUBLE_BITS_X86_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_LONG_DOUBLE_BITS_X86_H

#include "FPBits.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

template <unsigned Width> struct Padding;

// i386 padding.
template <> struct Padding<4> { static constexpr unsigned value = 16; };

// x86_64 padding.
template <> struct Padding<8> { static constexpr unsigned value = 48; };

template <> union FPBits<long double> {
  using UIntType = __uint128_t;

  static constexpr int exponentBias = 0x3FFF;
  static constexpr int maxExponent = 0x7FFF;
  static constexpr UIntType minSubnormal = UIntType(1);
  // Subnormal numbers include the implicit bit in x86 long double formats.
  static constexpr UIntType maxSubnormal =
      (UIntType(1) << (MantissaWidth<long double>::value)) - 1;
  static constexpr UIntType minNormal =
      (UIntType(3) << MantissaWidth<long double>::value);
  static constexpr UIntType maxNormal =
      ((UIntType(maxExponent) - 1) << (MantissaWidth<long double>::value + 1)) |
      (UIntType(1) << MantissaWidth<long double>::value) | maxSubnormal;

  using FloatProp = FloatProperties<long double>;

  UIntType bits;

  void setMantissa(UIntType mantVal) {
    mantVal &= (FloatProp::mantissaMask);
    bits &= ~(FloatProp::mantissaMask);
    bits |= mantVal;
  }

  UIntType getMantissa() const { return bits & FloatProp::mantissaMask; }

  void setUnbiasedExponent(UIntType expVal) {
    expVal = (expVal << (FloatProp::bitWidth - 1 - FloatProp::exponentWidth)) &
             FloatProp::exponentMask;
    bits &= ~(FloatProp::exponentMask);
    bits |= expVal;
  }

  uint16_t getUnbiasedExponent() const {
    return uint16_t((bits & FloatProp::exponentMask) >>
                    (FloatProp::bitWidth - 1 - FloatProp::exponentWidth));
  }

  void setImplicitBit(bool implicitVal) {
    bits &= ~(UIntType(1) << FloatProp::mantissaWidth);
    bits |= (UIntType(implicitVal) << FloatProp::mantissaWidth);
  }

  bool getImplicitBit() const {
    return ((bits & (UIntType(1) << FloatProp::mantissaWidth)) >>
            FloatProp::mantissaWidth);
  }

  void setSign(bool signVal) {
    bits &= ~(FloatProp::signMask);
    UIntType sign1 = UIntType(signVal) << (FloatProp::bitWidth - 1);
    bits |= sign1;
  }

  bool getSign() const {
    return ((bits & FloatProp::signMask) >> (FloatProp::bitWidth - 1));
  }

  long double val;

  FPBits() : bits(0) {}

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<long double, XType>::Value, int> = 0>
  explicit FPBits(XType x) : val(x) {}

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<XType, UIntType>::Value, int> = 0>
  explicit FPBits(XType x) : bits(x) {}

  operator long double() { return val; }

  UIntType uintval() {
    // We zero the padding bits as they can contain garbage.
    static constexpr UIntType mask =
        (UIntType(1) << (sizeof(long double) * 8 -
                         Padding<sizeof(uintptr_t)>::value)) -
        1;
    return bits & mask;
  }

  int getExponent() const {
    if (getUnbiasedExponent() == 0)
      return int(1) - exponentBias;
    return int(getUnbiasedExponent()) - exponentBias;
  }

  bool isZero() const {
    return getUnbiasedExponent() == 0 && getMantissa() == 0 &&
           getImplicitBit() == 0;
  }

  bool isInf() const {
    return getUnbiasedExponent() == maxExponent && getMantissa() == 0 &&
           getImplicitBit() == 1;
  }

  bool isNaN() const {
    if (getUnbiasedExponent() == maxExponent) {
      return (getImplicitBit() == 0) || getMantissa() != 0;
    } else if (getUnbiasedExponent() != 0) {
      return getImplicitBit() == 0;
    }
    return false;
  }

  bool isInfOrNaN() const {
    return (getUnbiasedExponent() == maxExponent) ||
           (getUnbiasedExponent() != 0 && getImplicitBit() == 0);
  }

  // Methods below this are used by tests.

  static FPBits<long double> zero() { return FPBits<long double>(0.0l); }

  static FPBits<long double> negZero() {
    FPBits<long double> bits(0.0l);
    bits.setSign(1);
    return bits;
  }

  static FPBits<long double> inf() {
    FPBits<long double> bits(0.0l);
    bits.setUnbiasedExponent(maxExponent);
    bits.setImplicitBit(1);
    return bits;
  }

  static FPBits<long double> negInf() {
    FPBits<long double> bits(0.0l);
    bits.setUnbiasedExponent(maxExponent);
    bits.setImplicitBit(1);
    bits.setSign(1);
    return bits;
  }

  static long double buildNaN(UIntType v) {
    FPBits<long double> bits(0.0l);
    bits.setUnbiasedExponent(maxExponent);
    bits.setImplicitBit(1);
    bits.setMantissa(v);
    return bits;
  }
};

static_assert(
    sizeof(FPBits<long double>) == sizeof(long double),
    "Internal long double representation does not match the machine format.");

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_LONG_DOUBLE_BITS_X86_H
