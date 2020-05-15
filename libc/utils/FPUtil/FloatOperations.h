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

// Return the zero adjusted exponent value of x.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
int getExponent(T x) {
  using Properties = FloatProperties<T>;
  using BitsType = typename Properties::BitsType;
  BitsType bits = absBits(x);
  int e = (bits >> Properties::mantissaWidth); // Shift out the mantissa.
  e -= Properties::exponentOffset;             // Zero adjust.
  return e;
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

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_FLOAT_OPERATIONS_H
