//===-- Utils for abs and friends -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H

#include "src/__support/CPP/type_traits.h"

namespace __llvm_libc {

template <typename T>
static constexpr cpp::enable_if_t<cpp::is_integral_v<T>, T> integer_abs(T n) {
  return (n < 0) ? -n : n;
}

template <typename T>
static constexpr cpp::enable_if_t<cpp::is_integral_v<T>, void>
integer_rem_quo(T x, T y, T &quot, T &rem) {
  quot = x / y;
  rem = x % y;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDLIB_ABS_UTILS_H
