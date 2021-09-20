//===-- Utils for abs and friends -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {

template <typename T>
static constexpr cpp::EnableIfType<cpp::IsIntegral<T>::Value, T>
integerAbs(T n) {
  return (n < 0) ? -n : n;
}

template <typename T>
static constexpr cpp::EnableIfType<cpp::IsIntegral<T>::Value, void>
integerRemQuo(T x, T y, T &quot, T &rem) {
  quot = x / y;
  rem = x % y;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H
