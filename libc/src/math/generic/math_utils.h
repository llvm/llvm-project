//===-- Collection of utils for implementing math functions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GENERIC_MATH_UTILS_H
#define LLVM_LIBC_SRC_MATH_GENERIC_MATH_UTILS_H

#include "include/llvm-libc-macros/math-macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

#include <stdint.h>

// TODO: evaluate which functions from this file are actually used.

namespace LIBC_NAMESPACE {

// TODO: Remove this, or move it to exp_utils.cpp which is its only user.
LIBC_INLINE double as_double(uint64_t x) { return cpp::bit_cast<double>(x); }

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
    libc_errno = err;
  return x;
}

template <typename T> LIBC_INLINE void force_eval(T x) {
  volatile T y LIBC_UNUSED = x;
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

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_GENERIC_MATH_UTILS_H
