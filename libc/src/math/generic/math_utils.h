//===-- Collection of utils for implementing math functions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_MATH_UTILS_H
#define LLVM_LIBC_SRC_MATH_MATH_UTILS_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include <errno.h>
#include <math.h>

#include <stdint.h>

namespace __llvm_libc {

LIBC_INLINE uint32_t as_uint32_bits(float x) {
  return cpp::bit_cast<uint32_t>(x);
}

LIBC_INLINE uint64_t as_uint64_bits(double x) {
  return cpp::bit_cast<uint64_t>(x);
}

LIBC_INLINE float as_float(uint32_t x) { return cpp::bit_cast<float>(x); }

LIBC_INLINE double as_double(uint64_t x) { return cpp::bit_cast<double>(x); }

LIBC_INLINE uint32_t top12_bits(float x) { return as_uint32_bits(x) >> 20; }

LIBC_INLINE uint32_t top12_bits(double x) { return as_uint64_bits(x) >> 52; }

// Values to trigger underflow and overflow.
template <typename T> struct XFlowValues;

template <> struct XFlowValues<float> {
  static const float OVERFLOW_VALUE;
  static const float UNDERFLOW_VALUE;
  static const float MAY_UNDERFLOW_VALUE;
};

template <> struct XFlowValues<double> {
  static const double OVERFLOW_VALUE;
  static const double UNDERFLOW_VALUE;
  static const double MAY_UNDERFLOW_VALUE;
};

template <typename T> LIBC_INLINE T with_errno(T x, int err) {
  if (math_errhandling & MATH_ERRNO)
    errno = err;
  return x;
}

template <typename T> LIBC_INLINE void force_eval(T x) {
  volatile T y UNUSED = x;
}

template <typename T> LIBC_INLINE T opt_barrier(T x) {
  volatile T y = x;
  return y;
}

template <typename T> struct IsFloatOrDouble {
  static constexpr bool
      Value = // NOLINT so that this Value can match the ones for IsSame
      cpp::is_same_v<T, float> || cpp::is_same_v<T, double>;
};

template <typename T>
using EnableIfFloatOrDouble = cpp::enable_if_t<IsFloatOrDouble<T>::Value, int>;

template <typename T, EnableIfFloatOrDouble<T> = 0>
T xflow(uint32_t sign, T y) {
  // Underflow happens when two extremely small values are multiplied.
  // Likewise, overflow happens when two large values are multiplied.
  y = opt_barrier(sign ? -y : y) * y;
  return with_errno(y, ERANGE);
}

template <typename T, EnableIfFloatOrDouble<T> = 0> T overflow(uint32_t sign) {
  return xflow(sign, XFlowValues<T>::OVERFLOW_VALUE);
}

template <typename T, EnableIfFloatOrDouble<T> = 0> T underflow(uint32_t sign) {
  return xflow(sign, XFlowValues<T>::UNDERFLOW_VALUE);
}

template <typename T, EnableIfFloatOrDouble<T> = 0>
T may_underflow(uint32_t sign) {
  return xflow(sign, XFlowValues<T>::MAY_UNDERFLOW_VALUE);
}

template <typename T, EnableIfFloatOrDouble<T> = 0>
LIBC_INLINE constexpr float invalid(T x) {
  T y = (x - x) / (x - x);
  return isnan(x) ? y : with_errno(y, EDOM);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_MATH_UTILS_H
