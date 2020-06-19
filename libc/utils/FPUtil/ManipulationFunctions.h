//===-- Floating-point manipulation functions -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitPatterns.h"
#include "ClassificationFunctions.h"
#include "FloatOperations.h"
#include "FloatProperties.h"
#include "NearestIntegerOperations.h"

#include "utils/CPP/TypeTraits.h"

#ifndef LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H
#define LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H

namespace __llvm_libc {
namespace fputil {

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T frexp(T x, int &exp) {
  using Properties = FloatProperties<T>;
  using BitsType = typename Properties::BitsType;

  auto bits = valueAsBits(x);
  if (bitsAreInfOrNaN(bits))
    return x;
  if (bitsAreZero(bits)) {
    exp = 0;
    return x;
  }

  exp = getExponentFromBits(bits) + 1;

  static constexpr BitsType resultExponent =
      Properties::exponentOffset - BitsType(1);
  // Capture the sign and mantissa part.
  bits &= (Properties::mantissaMask | Properties::signMask);
  // Insert the new exponent.
  bits |= (resultExponent << Properties::mantissaWidth);

  return valueFromBits(bits);
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T modf(T x, T &iptr) {
  auto bits = valueAsBits(x);
  if (bitsAreZero(bits) || bitsAreNaN(bits)) {
    iptr = x;
    return x;
  } else if (bitsAreInf(bits)) {
    iptr = x;
    return bits & FloatProperties<T>::signMask
               ? valueFromBits(BitPatterns<T>::negZero)
               : valueFromBits(BitPatterns<T>::zero);
  } else {
    iptr = trunc(x);
    if (x == iptr) {
      // If x is already an integer value, then return zero with the right
      // sign.
      return bits & FloatProperties<T>::signMask
                 ? valueFromBits(BitPatterns<T>::negZero)
                 : valueFromBits(BitPatterns<T>::zero);
    } else {
      return x - iptr;
    }
  }
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T copysign(T x, T y) {
  constexpr auto signMask = FloatProperties<T>::signMask;
  auto xbits = valueAsBits(x);
  auto ybits = valueAsBits(y);
  return valueFromBits((xbits & ~signMask) | (ybits & signMask));
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T logb(T x) {
  auto bits = valueAsBits(x);
  if (bitsAreZero(bits)) {
    // TODO(Floating point exception): Raise div-by-zero exception.
    // TODO(errno): POSIX requires setting errno to ERANGE.
    return valueFromBits(BitPatterns<T>::negInf);
  } else if (bitsAreInf(bits)) {
    return valueFromBits(BitPatterns<T>::inf);
  } else if (bitsAreNaN(bits)) {
    return x;
  } else {
    return getExponentFromBits(bits);
  }
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_MANIPULATION_FUNCTIONS_H
