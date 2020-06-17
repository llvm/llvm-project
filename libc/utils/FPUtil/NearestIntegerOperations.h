//===-- Nearest integer floating-point operations ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_NEAREST_INTEGER_OPERATIONS_H
#define LLVM_LIBC_UTILS_FPUTIL_NEAREST_INTEGER_OPERATIONS_H

#include "ClassificationFunctions.h"
#include "FloatOperations.h"
#include "FloatProperties.h"

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T trunc(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename FloatProperties<T>::BitsType;

  BitsType bits = valueAsBits(x);

  // If x is infinity, NaN or zero, return it.
  if (bitsAreInfOrNaN(bits) || bitsAreZero(bits))
    return x;

  int exponent = getExponentFromBits(bits);

  // If the exponent is greater than the most negative mantissa
  // exponent, then x is already an integer.
  if (exponent >= static_cast<int>(Properties::mantissaWidth))
    return x;

  // If the exponent is such that abs(x) is less than 1, then return 0.
  if (exponent <= -1) {
    if (Properties::signMask & bits)
      return T(-0.0);
    else
      return T(0.0);
  }

  uint32_t trimSize = Properties::mantissaWidth - exponent;
  return valueFromBits((bits >> trimSize) << trimSize);
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T ceil(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename FloatProperties<T>::BitsType;

  BitsType bits = valueAsBits(x);

  // If x is infinity NaN or zero, return it.
  if (bitsAreInfOrNaN(bits) || bitsAreZero(bits))
    return x;

  bool isNeg = bits & Properties::signMask;
  int exponent = getExponentFromBits(bits);

  // If the exponent is greater than the most negative mantissa
  // exponent, then x is already an integer.
  if (exponent >= static_cast<int>(Properties::mantissaWidth))
    return x;

  if (exponent <= -1) {
    if (isNeg)
      return T(-0.0);
    else
      return T(1.0);
  }

  uint32_t trimSize = Properties::mantissaWidth - exponent;
  // If x is already an integer, return it.
  if ((bits << (Properties::bitWidth - trimSize)) == 0)
    return x;

  BitsType truncBits = (bits >> trimSize) << trimSize;
  T truncValue = valueFromBits(truncBits);

  // If x is negative, the ceil operation is equivalent to the trunc operation.
  if (isNeg)
    return truncValue;

  return truncValue + T(1.0);
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T floor(T x) {
  auto bits = valueAsBits(x);
  if (FloatProperties<T>::signMask & bits) {
    return -ceil(-x);
  } else {
    return trunc(x);
  }
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T round(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename FloatProperties<T>::BitsType;

  BitsType bits = valueAsBits(x);

  // If x is infinity, NaN or zero, return it.
  if (bitsAreInfOrNaN(bits) || bitsAreZero(bits))
    return x;

  bool isNeg = bits & Properties::signMask;
  int exponent = getExponentFromBits(bits);

  // If the exponent is greater than the most negative mantissa
  // exponent, then x is already an integer.
  if (exponent >= static_cast<int>(Properties::mantissaWidth))
    return x;

  if (exponent == -1) {
    // Absolute value of x is greater than equal to 0.5 but less than 1.
    if (isNeg)
      return T(-1.0);
    else
      return T(1.0);
  }

  if (exponent <= -2) {
    // Absolute value of x is less than 0.5.
    if (isNeg)
      return T(-0.0);
    else
      return T(0.0);
  }

  uint32_t trimSize = Properties::mantissaWidth - exponent;
  // If x is already an integer, return it.
  if ((bits << (Properties::bitWidth - trimSize)) == 0)
    return x;

  BitsType truncBits = (bits >> trimSize) << trimSize;
  T truncValue = valueFromBits(truncBits);

  if ((bits & (BitsType(1) << (trimSize - 1))) == 0) {
    // Franctional part is less than 0.5 so round value is the
    // same as the trunc value.
    return truncValue;
  }

  if (isNeg)
    return truncValue - T(1.0);
  else
    return truncValue + T(1.0);
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_NEAREST_INTEGER_OPERATIONS_H
