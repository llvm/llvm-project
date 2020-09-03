//===-- Floating point classification functions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_CLASSIFICATION_FUNCTIONS_H
#define LLVM_LIBC_UTILS_FPUTIL_CLASSIFICATION_FUNCTIONS_H

#include "BitPatterns.h"
#include "FloatOperations.h"
#include "FloatProperties.h"

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

template <typename BitsType> static inline bool bitsAreInf(BitsType bits) {
  using FPType = typename FloatType<BitsType>::Type;
  return ((bits & BitPatterns<FPType>::inf) == BitPatterns<FPType>::inf) &&
         ((bits & FloatProperties<FPType>::mantissaMask) == 0);
}

// Return true if x is infinity (positive or negative.)
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline bool isInf(T x) {
  return bitsAreInf(valueAsBits(x));
}

template <typename BitsType> static inline bool bitsAreNaN(BitsType bits) {
  using FPType = typename FloatType<BitsType>::Type;
  return ((bits & BitPatterns<FPType>::inf) == BitPatterns<FPType>::inf) &&
         ((bits & FloatProperties<FPType>::mantissaMask) != 0);
}

// Return true if x is a NAN (quiet or signalling.)
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline bool isNaN(T x) {
  return bitsAreNaN(valueAsBits(x));
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

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_CLASSIFICATION_FUNCTIONS_H
