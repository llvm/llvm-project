//===-- Abstract class for bit manipulation of float numbers. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_H

#include "PlatformDefs.h"

#include "utils/CPP/TypeTraits.h"

#include "FloatProperties.h"
#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

template <typename T> struct MantissaWidth {
  static constexpr unsigned value = FloatProperties<T>::mantissaWidth;
};

template <typename T> struct ExponentWidth {
  static constexpr unsigned value = FloatProperties<T>::exponentWidth;
};

// A generic class to represent single precision, double precision, and quad
// precision IEEE 754 floating point formats.
// On most platforms, the 'float' type corresponds to single precision floating
// point numbers, the 'double' type corresponds to double precision floating
// point numers, and the 'long double' type corresponds to the quad precision
// floating numbers. On x86 platforms however, the 'long double' type maps to
// an x87 floating point format. This format is an IEEE 754 extension format.
// It is handled as an explicit specialization of this class.
template <typename T> union FPBits {
  static_assert(cpp::IsFloatingPointType<T>::Value,
                "FPBits instantiated with invalid type.");

  // Reinterpreting bits as an integer value and interpreting the bits of an
  // integer value as a floating point value is used in tests. So, a convenient
  // type is provided for such reinterpretations.
  using FloatProp = FloatProperties<T>;
  // TODO: Change UintType name to BitsType for consistency.
  using UIntType = typename FloatProp::BitsType;

  UIntType bits;

  void setMantissa(UIntType mantVal) {
    mantVal &= (FloatProp::mantissaMask);
    bits &= ~(FloatProp::mantissaMask);
    bits |= mantVal;
  }

  UIntType getMantissa() const { return bits & FloatProp::mantissaMask; }

  void setUnbiasedExponent(UIntType expVal) {
    expVal = (expVal << (FloatProp::mantissaWidth)) & FloatProp::exponentMask;
    bits &= ~(FloatProp::exponentMask);
    bits |= expVal;
  }

  uint16_t getUnbiasedExponent() const {
    return uint16_t((bits & FloatProp::exponentMask) >>
                    (FloatProp::mantissaWidth));
  }

  void setSign(bool signVal) {
    bits &= ~(FloatProp::signMask);
    UIntType sign = UIntType(signVal) << (FloatProp::bitWidth - 1);
    bits |= sign;
  }

  bool getSign() const {
    return ((bits & FloatProp::signMask) >> (FloatProp::bitWidth - 1));
  }
  T val;

  static_assert(sizeof(T) == sizeof(UIntType),
                "Data type and integral representation have different sizes.");

  static constexpr int exponentBias = (1 << (ExponentWidth<T>::value - 1)) - 1;
  static constexpr int maxExponent = (1 << ExponentWidth<T>::value) - 1;

  static constexpr UIntType minSubnormal = UIntType(1);
  static constexpr UIntType maxSubnormal =
      (UIntType(1) << MantissaWidth<T>::value) - 1;
  static constexpr UIntType minNormal =
      (UIntType(1) << MantissaWidth<T>::value);
  static constexpr UIntType maxNormal =
      ((UIntType(maxExponent) - 1) << MantissaWidth<T>::value) | maxSubnormal;

  // We don't want accidental type promotions/conversions so we require exact
  // type match.
  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<T, XType>::Value, int> = 0>
  explicit FPBits(XType x) : val(x) {}

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<XType, UIntType>::Value, int> = 0>
  explicit FPBits(XType x) : bits(x) {}

  FPBits() : bits(0) {}

  explicit operator T() { return val; }

  UIntType uintval() const { return bits; }

  int getExponent() const { return int(getUnbiasedExponent()) - exponentBias; }

  bool isZero() const {
    return getMantissa() == 0 && getUnbiasedExponent() == 0;
  }

  bool isInf() const {
    return getMantissa() == 0 && getUnbiasedExponent() == maxExponent;
  }

  bool isNaN() const {
    return getUnbiasedExponent() == maxExponent && getMantissa() != 0;
  }

  bool isInfOrNaN() const { return getUnbiasedExponent() == maxExponent; }

  static FPBits<T> zero() { return FPBits(); }

  static FPBits<T> negZero() {
    return FPBits(UIntType(1) << (sizeof(UIntType) * 8 - 1));
  }

  static FPBits<T> inf() {
    FPBits<T> bits;
    bits.setUnbiasedExponent(maxExponent);
    return bits;
  }

  static FPBits<T> negInf() {
    FPBits<T> bits = inf();
    bits.setSign(1);
    return bits;
  }

  static T buildNaN(UIntType v) {
    FPBits<T> bits = inf();
    bits.setMantissa(v);
    return T(bits);
  }
};

} // namespace fputil
} // namespace __llvm_libc

#ifdef SPECIAL_X86_LONG_DOUBLE
#include "src/__support/FPUtil/LongDoubleBitsX86.h"
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FP_BITS_H
