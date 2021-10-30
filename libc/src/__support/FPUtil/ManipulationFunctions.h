//===-- Floating-point manipulation functions -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_MANIPULATION_FUNCTIONS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_MANIPULATION_FUNCTIONS_H

#include "FPBits.h"
#include "NearestIntegerOperations.h"
#include "NormalFloat.h"
#include "PlatformDefs.h"

#include "src/__support/CPP/TypeTraits.h"

#include <limits.h>
#include <math.h>

namespace __llvm_libc {
namespace fputil {

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T frexp(T x, int &exp) {
  FPBits<T> bits(x);
  if (bits.isInfOrNaN())
    return x;
  if (bits.isZero()) {
    exp = 0;
    return x;
  }

  NormalFloat<T> normal(bits);
  exp = normal.exponent + 1;
  normal.exponent = -1;
  return normal;
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T modf(T x, T &iptr) {
  FPBits<T> bits(x);
  if (bits.isZero() || bits.isNaN()) {
    iptr = x;
    return x;
  } else if (bits.isInf()) {
    iptr = x;
    return bits.getSign() ? T(FPBits<T>::negZero()) : T(FPBits<T>::zero());
  } else {
    iptr = trunc(x);
    if (x == iptr) {
      // If x is already an integer value, then return zero with the right
      // sign.
      return bits.getSign() ? T(FPBits<T>::negZero()) : T(FPBits<T>::zero());
    } else {
      return x - iptr;
    }
  }
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T copysign(T x, T y) {
  FPBits<T> xbits(x);
  xbits.setSign(FPBits<T>(y).getSign());
  return T(xbits);
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline int ilogb(T x) {
  // TODO: Raise appropriate floating point exceptions and set errno to the
  // an appropriate error value wherever relevant.
  FPBits<T> bits(x);
  if (bits.isZero()) {
    return FP_ILOGB0;
  } else if (bits.isNaN()) {
    return FP_ILOGBNAN;
  } else if (bits.isInf()) {
    return INT_MAX;
  }

  NormalFloat<T> normal(bits);
  // The C standard does not specify the return value when an exponent is
  // out of int range. However, XSI conformance required that INT_MAX or
  // INT_MIN are returned.
  // NOTE: It is highly unlikely that exponent will be out of int range as
  // the exponent is only 15 bits wide even for the 128-bit floating point
  // format.
  if (normal.exponent > INT_MAX)
    return INT_MAX;
  else if (normal.exponent < INT_MIN)
    return INT_MIN;
  else
    return normal.exponent;
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T logb(T x) {
  FPBits<T> bits(x);
  if (bits.isZero()) {
    // TODO(Floating point exception): Raise div-by-zero exception.
    // TODO(errno): POSIX requires setting errno to ERANGE.
    return T(FPBits<T>::negInf());
  } else if (bits.isNaN()) {
    return x;
  } else if (bits.isInf()) {
    // Return positive infinity.
    return T(FPBits<T>::inf());
  }

  NormalFloat<T> normal(bits);
  return normal.exponent;
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T ldexp(T x, int exp) {
  FPBits<T> bits(x);
  if (bits.isZero() || bits.isInfOrNaN() || exp == 0)
    return x;

  // NormalFloat uses int32_t to store the true exponent value. We should ensure
  // that adding |exp| to it does not lead to integer rollover. But, if |exp|
  // value is larger the exponent range for type T, then we can return infinity
  // early. Because the result of the ldexp operation can be a subnormal number,
  // we need to accommodate the (mantissaWidht + 1) worth of shift in
  // calculating the limit.
  int expLimit = FPBits<T>::maxExponent + MantissaWidth<T>::value + 1;
  if (exp > expLimit)
    return bits.getSign() ? T(FPBits<T>::negInf()) : T(FPBits<T>::inf());

  // Similarly on the negative side we return zero early if |exp| is too small.
  if (exp < -expLimit)
    return bits.getSign() ? T(FPBits<T>::negZero()) : T(FPBits<T>::zero());

  // For all other values, NormalFloat to T conversion handles it the right way.
  NormalFloat<T> normal(bits);
  normal.exponent += exp;
  return normal;
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T nextafter(T from, T to) {
  FPBits<T> fromBits(from);
  if (fromBits.isNaN())
    return from;

  FPBits<T> toBits(to);
  if (toBits.isNaN())
    return to;

  if (from == to)
    return to;

  using UIntType = typename FPBits<T>::UIntType;
  UIntType intVal = fromBits.uintval();
  UIntType signMask = (UIntType(1) << (sizeof(T) * 8 - 1));
  if (from != T(0.0)) {
    if ((from < to) == (from > T(0.0))) {
      ++intVal;
    } else {
      --intVal;
    }
  } else {
    intVal = (toBits.uintval() & signMask) + UIntType(1);
  }

  return *reinterpret_cast<T *>(&intVal);
  // TODO: Raise floating point exceptions as required by the standard.
}

} // namespace fputil
} // namespace __llvm_libc

#ifdef SPECIAL_X86_LONG_DOUBLE
#include "NextAfterLongDoubleX86.h"
#endif // SPECIAL_X86_LONG_DOUBLE

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_MANIPULATION_FUNCTIONS_H
