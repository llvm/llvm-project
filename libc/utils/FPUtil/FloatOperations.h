//===-- Common operations on floating point numbers -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_FLOAT_OPERATIONS_H
#define LLVM_LIBC_UTILS_FPUTIL_FLOAT_OPERATIONS_H

#include "BitPatterns.h"
#include "FloatProperties.h"

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

// Return the bits of a float value.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline typename FloatProperties<T>::BitsType valueAsBits(T x) {
  using BitsType = typename FloatProperties<T>::BitsType;
  return *reinterpret_cast<BitsType *>(&x);
}

// Return the float value from bits.
template <typename BitsType,
          cpp::EnableIfType<
              cpp::IsFloatingPointType<FloatTypeT<BitsType>>::Value, int> = 0>
static inline FloatTypeT<BitsType> valueFromBits(BitsType bits) {
  return *reinterpret_cast<FloatTypeT<BitsType> *>(&bits);
}

// Return the bits of abs(x).
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline typename FloatProperties<T>::BitsType absBits(T x) {
  return valueAsBits(x) & (~FloatProperties<T>::signMask);
}

template <typename BitsType>
static inline int getExponentFromBits(BitsType bits) {
  using FPType = typename FloatType<BitsType>::Type;
  using Properties = FloatProperties<FPType>;
  bits &= Properties::exponentMask;
  int e = (bits >> Properties::mantissaWidth); // Shift out the mantissa.
  e -= Properties::exponentOffset;             // Zero adjust.
  return e;
}

// Return the zero adjusted exponent value of x.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline int getExponent(T x) {
  return getExponentFromBits(valueAsBits(x));
}

// Return true if x is infinity (positive or negative.)
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline bool isInf(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename FloatProperties<T>::BitsType;
  BitsType bits = valueAsBits(x);
  return ((bits & BitPatterns<T>::inf) == BitPatterns<T>::inf) &&
         ((bits & Properties::mantissaMask) == 0);
}

// Return true if x is a NAN (quiet or signalling.)
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline bool isNaN(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename FloatProperties<T>::BitsType;
  BitsType bits = valueAsBits(x);
  return ((bits & BitPatterns<T>::inf) == BitPatterns<T>::inf) &&
         ((bits & Properties::mantissaMask) != 0);
}

template <typename BitsType> static inline bool bitsAreInfOrNaN(BitsType bits) {
  using FPType = typename FloatType<BitsType>::Type;
  return (bits & BitPatterns<FPType>::inf) == BitPatterns<FPType>::inf;
}

template <typename BitsType> static inline bool bitsAreZero(BitsType bits) {
  using FPType = typename FloatType<BitsType>::Type;
  return (bits == BitPatterns<FPType>::zero) ||
         (bits == BitPatterns<FPType>::negZero);
}

// Return true if x is any kind of NaN or infinity.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline bool isInfOrNaN(T x) {
  return bitsAreInfOrNaN(valueAsBits(x));
}

// Return true if x is a quiet NAN.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline bool isQuietNaN(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename FloatProperties<T>::BitsType;
  BitsType bits = valueAsBits(x);
  return ((bits & BitPatterns<T>::inf) == BitPatterns<T>::inf) &&
         ((bits & Properties::quietNaNMask) != 0);
}

// Return true if x is a quiet NAN with sign bit set.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline bool isNegativeQuietNaN(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename FloatProperties<T>::BitsType;
  BitsType bits = valueAsBits(x);
  return ((bits & BitPatterns<T>::negInf) == BitPatterns<T>::negInf) &&
         ((bits & Properties::quietNaNMask) != 0);
}

// Return the absolute value of x.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T abs(T x) {
  return valueFromBits(absBits(x));
}

// Return the trucated value of x. If x is non-negative, then the return value
// is greatest integer less than or equal to x. Otherwise, return the smallest
// integer greater than or equal to x. That is, return the integer value rounded
// toward zero.
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

#endif // LLVM_LIBC_UTILS_FPUTIL_FLOAT_OPERATIONS_H
