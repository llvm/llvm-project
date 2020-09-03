//===-- Floating-point manipulation functions -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPBits.h"
#include "NearestIntegerOperations.h"

#include "utils/CPP/TypeTraits.h"

#ifndef LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H
#define LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H

namespace __llvm_libc {
namespace fputil {

#if defined(__x86_64__) || defined(__i386__)
template <typename T> struct Standard754Type {
  static constexpr bool Value =
      cpp::IsSame<float, cpp::RemoveCVType<T>>::Value ||
      cpp::IsSame<double, cpp::RemoveCVType<T>>::Value;
};
#else
template <typename T> struct Standard754Type {
  static constexpr bool Value = cpp::IsFloatingPointType<T>::Value;
};
#endif

template <typename T> static inline T frexp_impl(FPBits<T> &bits, int &exp) {
  exp = bits.getExponent() + 1;
  static constexpr uint16_t resultExponent = FPBits<T>::exponentBias - 1;
  bits.exponent = resultExponent;
  return bits;
}

template <typename T, cpp::EnableIfType<Standard754Type<T>::Value, int> = 0>
static inline T frexp(T x, int &exp) {
  FPBits<T> bits(x);
  if (bits.isInfOrNaN())
    return x;
  if (bits.isZero()) {
    exp = 0;
    return x;
  }

  return frexp_impl(bits, exp);
}

#if defined(__x86_64__) || defined(__i386__)
static inline long double frexp(long double x, int &exp) {
  FPBits<long double> bits(x);
  if (bits.isInfOrNaN())
    return x;
  if (bits.isZero()) {
    exp = 0;
    return x;
  }

  if (bits.exponent != 0 || bits.implicitBit == 1)
    return frexp_impl(bits, exp);

  exp = bits.getExponent();
  int shiftCount = 0;
  uint64_t fullMantissa = *reinterpret_cast<uint64_t *>(&bits);
  static constexpr uint64_t msBitMask = uint64_t(1) << 63;
  for (; (fullMantissa & msBitMask) == uint64_t(0);
       fullMantissa <<= 1, ++shiftCount) {
    // This for loop will terminate as fullMantissa is != 0. If it were 0,
    // then x will be NaN and handled before control reaches here.
    // When the loop terminates, fullMantissa will represent the full mantissa
    // of a normal long double value. That is, the implicit bit has the value
    // of 1.
  }

  exp = exp - shiftCount + 1;
  *reinterpret_cast<uint64_t *>(&bits) = fullMantissa;
  bits.exponent = FPBits<long double>::exponentBias - 1;
  return bits;
}
#endif

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T modf(T x, T &iptr) {
  FPBits<T> bits(x);
  if (bits.isZero() || bits.isNaN()) {
    iptr = x;
    return x;
  } else if (bits.isInf()) {
    iptr = x;
    return bits.sign ? FPBits<T>::negZero() : FPBits<T>::zero();
  } else {
    iptr = trunc(x);
    if (x == iptr) {
      // If x is already an integer value, then return zero with the right
      // sign.
      return bits.sign ? FPBits<T>::negZero() : FPBits<T>::zero();
    } else {
      return x - iptr;
    }
  }
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T copysign(T x, T y) {
  FPBits<T> xbits(x);
  xbits.sign = FPBits<T>(y).sign;
  return xbits;
}

template <typename T> static inline T logb_impl(const FPBits<T> &bits) {
  return bits.getExponent();
}

template <typename T, cpp::EnableIfType<Standard754Type<T>::Value, int> = 0>
static inline T logb(T x) {
  FPBits<T> bits(x);
  if (bits.isZero()) {
    // TODO(Floating point exception): Raise div-by-zero exception.
    // TODO(errno): POSIX requires setting errno to ERANGE.
    return FPBits<T>::negInf();
  } else if (bits.isNaN()) {
    return x;
  } else if (bits.isInf()) {
    // Return positive infinity.
    return FPBits<T>::inf();
  }

  return logb_impl(bits);
}

#if defined(__x86_64__) || defined(__i386__)
static inline long double logb(long double x) {
  FPBits<long double> bits(x);
  if (bits.isZero()) {
    // TODO(Floating point exception): Raise div-by-zero exception.
    // TODO(errno): POSIX requires setting errno to ERANGE.
    return FPBits<long double>::negInf();
  } else if (bits.isNaN()) {
    return x;
  } else if (bits.isInf()) {
    // Return positive infinity.
    return FPBits<long double>::inf();
  }

  if (bits.exponent != 0 || bits.implicitBit == 1)
    return logb_impl(bits);

  int exp = bits.getExponent();
  int shiftCount = 0;
  uint64_t fullMantissa = *reinterpret_cast<uint64_t *>(&bits);
  static constexpr uint64_t msBitMask = uint64_t(1) << 63;
  for (; (fullMantissa & msBitMask) == uint64_t(0);
       fullMantissa <<= 1, ++shiftCount) {
    // This for loop will terminate as fullMantissa is != 0. If it were 0,
    // then x will be NaN and handled before control reaches here.
    // When the loop terminates, fullMantissa will represent the full mantissa
    // of a normal long double value. That is, the implicit bit has the value
    // of 1.
  }

  return exp - shiftCount;
}
#endif

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H
