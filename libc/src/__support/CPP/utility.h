//===-- Analogous to <utility> ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"

namespace __llvm_libc::cpp {

template <typename T, T... Ints> struct integer_sequence {
  static_assert(is_integral_v<T>);
  template <T Next> using append = integer_sequence<T, Ints..., Next>;
};

namespace internal {

template <typename T, int N> struct make_integer_sequence {
  using type =
      typename make_integer_sequence<T, N - 1>::type::template append<N>;
};

template <typename T> struct make_integer_sequence<T, -1> {
  using type = integer_sequence<T>;
};

} // namespace internal

template <typename T, int N>
using make_integer_sequence =
    typename internal::make_integer_sequence<T, N - 1>::type;

template <typename T>
LIBC_INLINE constexpr T &&forward(typename remove_reference<T>::type &value) {
  return static_cast<T &&>(value);
}

template <typename T>
LIBC_INLINE constexpr T &&forward(typename remove_reference<T>::type &&value) {
  return static_cast<T &&>(value);
}

template <typename T>
LIBC_INLINE constexpr typename remove_reference<T>::type &&move(T &&value) {
  return static_cast<typename remove_reference<T>::type &&>(value);
}

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_H
