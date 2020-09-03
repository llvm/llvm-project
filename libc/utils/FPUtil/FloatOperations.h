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

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_FLOAT_OPERATIONS_H
