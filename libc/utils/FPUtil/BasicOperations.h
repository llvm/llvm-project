//===-- Basic operations on floating point numbers --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_BASIC_OPERATIONS_H
#define LLVM_LIBC_UTILS_FPUTIL_BASIC_OPERATIONS_H

#include "FPBits.h"

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T abs(T x) {
  FPBits<T> bits(x);
  bits.sign = 0;
  return T(bits);
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T fmin(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (bitx.isNaN()) {
    return y;
  } else if (bity.isNaN()) {
    return x;
  } else if (bitx.sign != bity.sign) {
    // To make sure that fmin(+0, -0) == -0 == fmin(-0, +0), whenever x and
    // y has different signs and both are not NaNs, we return the number
    // with negative sign.
    return (bitx.sign ? x : y);
  } else {
    return (x < y ? x : y);
  }
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T fmax(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (bitx.isNaN()) {
    return y;
  } else if (bity.isNaN()) {
    return x;
  } else if (bitx.sign != bity.sign) {
    // To make sure that fmax(+0, -0) == +0 == fmax(-0, +0), whenever x and
    // y has different signs and both are not NaNs, we return the number
    // with positive sign.
    return (bitx.sign ? y : x);
  } else {
    return (x > y ? x : y);
  }
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T fdim(T x, T y) {
  FPBits<T> bitx(x), bity(y);

  if (bitx.isNaN()) {
    return x;
  }

  if (bity.isNaN()) {
    return y;
  }

  return (x > y ? x - y : 0);
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_BASIC_OPERATIONS_H
