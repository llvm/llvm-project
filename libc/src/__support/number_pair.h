//===-- Utilities for pairs of numbers. -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_NUMBER_PAIR_H
#define LLVM_LIBC_SRC___SUPPORT_NUMBER_PAIR_H

#include "CPP/type_traits.h"
#include "named_pair.h"

#include <stddef.h>

namespace __llvm_libc {

DEFINE_NAMED_PAIR_TEMPLATE(NumberPair, lo, hi);

template <typename T>
cpp::enable_if_t<cpp::is_integral_v<T> && cpp::is_unsigned_v<T>, NumberPair<T>>
split(T a) {
  constexpr size_t HALF_BIT_WIDTH = sizeof(T) * 4;
  constexpr T LOWER_HALF_MASK = (T(1) << HALF_BIT_WIDTH) - T(1);
  NumberPair<T> result;
  result.lo = a & LOWER_HALF_MASK;
  result.hi = a >> HALF_BIT_WIDTH;
  return result;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_NUMBER_PAIR_H
