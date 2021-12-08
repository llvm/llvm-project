//===-- Collection of utils for implementing math functions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_MATH_UTILS_H
#define LLVM_LIBC_SRC_MATH_MATH_UTILS_H

#include "src/__support/CPP/TypeTraits.h"
#include "src/__support/common.h"
#include <errno.h>
#include <math.h>

#include <stdint.h>

namespace __llvm_libc {

static inline uint32_t as_uint32_bits(float x) {
  return *reinterpret_cast<uint32_t *>(&x);
}

static inline uint64_t as_uint64_bits(double x) {
  return *reinterpret_cast<uint64_t *>(&x);
}

static inline float as_float(uint32_t x) {
  return *reinterpret_cast<float *>(&x);
}

static inline double as_double(uint64_t x) {
  return *reinterpret_cast<double *>(&x);
}

static inline uint32_t top12_bits(float x) { return as_uint32_bits(x) >> 20; }

static inline uint32_t top12_bits(double x) { return as_uint64_bits(x) >> 52; }

// Values to trigger underflow and overflow.
template <typename T> struct XFlowValues;

template <> struct XFlowValues<float> {
  static const float overflow_value;
  static const float underflow_value;
  static const float may_underflow_value;
};

template <> struct XFlowValues<double> {
  static const double overflow_value;
  static const double underflow_value;
  static const double may_underflow_value;
};

template <typename T> static inline T with_errno(T x, int err) {
  if (math_errhandling & MATH_ERRNO)
    errno = err;
  return x;
}

template <typename T> static inline void force_eval(T x) {
  volatile T y UNUSED = x;
}

template <typename T> static inline T opt_barrier(T x) {
  volatile T y = x;
  return y;
}

template <typename T> struct IsFloatOrDouble {
  static constexpr bool
      Value = // NOLINT so that this Value can match the ones for IsSame
      cpp::IsSame<T, float>::Value || cpp::IsSame<T, double>::Value;
};

template <typename T>
using EnableIfFloatOrDouble = cpp::EnableIfType<IsFloatOrDouble<T>::Value, int>;

template <typename T, EnableIfFloatOrDouble<T> = 0>
T xflow(uint32_t sign, T y) {
  // Underflow happens when two extremely small values are multiplied.
  // Likewise, overflow happens when two large values are multiplied.
  y = opt_barrier(sign ? -y : y) * y;
  return with_errno(y, ERANGE);
}

template <typename T, EnableIfFloatOrDouble<T> = 0> T overflow(uint32_t sign) {
  return xflow(sign, XFlowValues<T>::overflow_value);
}

template <typename T, EnableIfFloatOrDouble<T> = 0> T underflow(uint32_t sign) {
  return xflow(sign, XFlowValues<T>::underflow_value);
}

template <typename T, EnableIfFloatOrDouble<T> = 0>
T may_underflow(uint32_t sign) {
  return xflow(sign, XFlowValues<T>::may_underflow_value);
}

template <typename T, EnableIfFloatOrDouble<T> = 0>
static inline constexpr float invalid(T x) {
  T y = (x - x) / (x - x);
  return isnan(x) ? y : with_errno(y, EDOM);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_MATH_UTILS_H
