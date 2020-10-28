//===-- Floating-point manipulation functions -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H
#define LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H

#include "FPBits.h"
#include "NearestIntegerOperations.h"
#include "NormalFloat.h"

#include "utils/CPP/TypeTraits.h"

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

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
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

  NormalFloat<T> normal(bits);
  return normal.exponent;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H
