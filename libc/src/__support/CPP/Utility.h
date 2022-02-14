//===-- Analogous to <utility> ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_H

#include "src/__support/CPP/TypeTraits.h"

namespace __llvm_libc::cpp {

template <typename T, T... Seq> struct IntegerSequence {
  static_assert(IsIntegral<T>::Value);
  template <T Next> using append = IntegerSequence<T, Seq..., Next>;
};

namespace internal {

template <typename T, int N> struct MakeIntegerSequence {
  using type = typename MakeIntegerSequence<T, N - 1>::type::template append<N>;
};

template <typename T> struct MakeIntegerSequence<T, -1> {
  using type = IntegerSequence<T>;
};

} // namespace internal

template <typename T, int N>
using MakeIntegerSequence =
    typename internal::MakeIntegerSequence<T, N - 1>::type;

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_H
