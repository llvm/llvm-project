//===-- Bit representation of x86 long double numbers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_LONG_DOUBLE_BITS_X86_H
#define LLVM_LIBC_UTILS_FPUTIL_LONG_DOUBLE_BITS_X86_H

#include "FPBits.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

template <> struct MantissaWidth<long double> {
  static constexpr unsigned value = 63;
};

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

  struct __attribute__((packed)) {
    UIntType mantissa : MantissaWidth<long double>::value;
    uint8_t implicitBit : 1;
    uint16_t exponent : ExponentWidth<long double>::value;
    uint8_t sign : 1;
    uint64_t padding : Padding<sizeof(uintptr_t)>::value;
  } encoding;
  UIntType integer;
  long double val;

  FPBits() : integer(0) {}

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<long double, XType>::Value, int> = 0>
  explicit FPBits<long double>(XType x) : val(x) {}

  template <typename XType,
            cpp::EnableIfType<cpp::IsSame<XType, UIntType>::Value, int> = 0>
  explicit FPBits(XType x) : integer(x) {}

  operator long double() { return val; }

  UIntType uintval() {
    // We zero the padding bits as they can contain garbage.
    static constexpr UIntType mask =
        (UIntType(1) << (sizeof(long double) * 8 -
                         Padding<sizeof(uintptr_t)>::value)) -
        1;
    return integer & mask;
  }

  int getExponent() const {
    if (encoding.exponent == 0)
      return int(1) - exponentBias;
    return int(encoding.exponent) - exponentBias;
  }

  bool isZero() const {
    return encoding.exponent == 0 && encoding.mantissa == 0 &&
           encoding.implicitBit == 0;
  }

  bool isInf() const {
    return encoding.exponent == maxExponent && encoding.mantissa == 0 &&
           encoding.implicitBit == 1;
  }

  bool isNaN() const {
    if (encoding.exponent == maxExponent) {
      return (encoding.implicitBit == 0) || encoding.mantissa != 0;
    } else if (encoding.exponent != 0) {
      return encoding.implicitBit == 0;
    }
    return false;
  }

  bool isInfOrNaN() const {
    return (encoding.exponent == maxExponent) ||
           (encoding.exponent != 0 && encoding.implicitBit == 0);
  }

  // Methods below this are used by tests.

  static FPBits<long double> zero() { return FPBits<long double>(0.0l); }

  static FPBits<long double> negZero() {
    FPBits<long double> bits(0.0l);
    bits.encoding.sign = 1;
    return bits;
  }

  static FPBits<long double> inf() {
    FPBits<long double> bits(0.0l);
    bits.encoding.exponent = maxExponent;
    bits.encoding.implicitBit = 1;
    return bits;
  }

  static FPBits<long double> negInf() {
    FPBits<long double> bits(0.0l);
    bits.encoding.exponent = maxExponent;
    bits.encoding.implicitBit = 1;
    bits.encoding.sign = 1;
    return bits;
  }

  static long double buildNaN(UIntType v) {
    FPBits<long double> bits(0.0l);
    bits.encoding.exponent = maxExponent;
    bits.encoding.implicitBit = 1;
    bits.encoding.mantissa = v;
    return bits;
  }
};

static_assert(
    sizeof(FPBits<long double>) == sizeof(long double),
    "Internal long double representation does not match the machine format.");

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_LONG_DOUBLE_BITS_X86_H
