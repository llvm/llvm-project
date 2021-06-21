//===-- Abstract class for bit manipulation of float numbers. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_FP_BITS_H
#define LLVM_LIBC_UTILS_FPUTIL_FP_BITS_H

#include "PlatformDefs.h"

#include "utils/CPP/TypeTraits.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

template <typename T> struct MantissaWidth {};
template <> struct MantissaWidth<float> {
  static constexpr unsigned value = 23;
};
template <> struct MantissaWidth<double> {
  static constexpr unsigned value = 52;
};

template <typename T> struct ExponentWidth {};
template <> struct ExponentWidth<float> {
  static constexpr unsigned value = 8;
};
template <> struct ExponentWidth<double> {
  static constexpr unsigned value = 11;
};
template <> struct ExponentWidth<long double> {
  static constexpr unsigned value = 15;
};

template <typename T> struct FPUIntType {};
template <> struct FPUIntType<float> { using Type = uint32_t; };
template <> struct FPUIntType<double> { using Type = uint64_t; };

#ifdef LONG_DOUBLE_IS_DOUBLE
template <> struct MantissaWidth<long double> {
  static constexpr unsigned value = MantissaWidth<double>::value;
};
template <> struct FPUIntType<long double> {
  using Type = FPUIntType<double>::Type;
};
#elif !defined(SPECIAL_X86_LONG_DOUBLE)
template <> struct MantissaWidth<long double> {
  static constexpr unsigned value = 112;
};
template <> struct FPUIntType<long double> { using Type = __uint128_t; };
#endif

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
  using UIntType = typename FPUIntType<T>::Type;

  struct __attribute__((packed)) {
    UIntType mantissa : MantissaWidth<T>::value;
    uint16_t exponent : ExponentWidth<T>::value;
    uint8_t sign : 1;
  } encoding;
  UIntType integer;
  T val;

  static_assert(sizeof(encoding) == sizeof(UIntType),
                "Encoding and integral representation have different sizes.");
  static_assert(sizeof(integer) == sizeof(UIntType),
                "Integral representation and value type have different sizes.");

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
  explicit FPBits(XType x) : integer(x) {}

  FPBits() : integer(0) {}

  explicit operator T() { return val; }

  UIntType uintval() const { return integer; }

  int getExponent() const { return int(encoding.exponent) - exponentBias; }

  bool isZero() const {
    return encoding.mantissa == 0 && encoding.exponent == 0;
  }

  bool isInf() const {
    return encoding.mantissa == 0 && encoding.exponent == maxExponent;
  }

  bool isNaN() const {
    return encoding.exponent == maxExponent && encoding.mantissa != 0;
  }

  bool isInfOrNaN() const { return encoding.exponent == maxExponent; }

  static FPBits<T> zero() { return FPBits(); }

  static FPBits<T> negZero() {
    return FPBits(UIntType(1) << (sizeof(UIntType) * 8 - 1));
  }

  static FPBits<T> inf() {
    FPBits<T> bits;
    bits.encoding.exponent = maxExponent;
    return bits;
  }

  static FPBits<T> negInf() {
    FPBits<T> bits = inf();
    bits.encoding.sign = 1;
    return bits;
  }

  static T buildNaN(UIntType v) {
    FPBits<T> bits = inf();
    bits.encoding.mantissa = v;
    return T(bits);
  }
};

} // namespace fputil
} // namespace __llvm_libc

#ifdef SPECIAL_X86_LONG_DOUBLE
#include "utils/FPUtil/LongDoubleBitsX86.h"
#endif

#endif // LLVM_LIBC_UTILS_FPUTIL_FP_BITS_H
