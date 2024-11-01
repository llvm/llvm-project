//===-- in_place utility ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_IN_PLACE_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_IN_PLACE_H

#include "src/__support/macros/attributes.h"

#include <stddef.h> // size_t

namespace __llvm_libc::cpp {

// in_place
struct in_place_t {
  explicit in_place_t() = default;
};
LIBC_INLINE_VAR constexpr in_place_t in_place{};

template <class T> struct in_place_type_t {
  explicit in_place_type_t() = default;
};
template <class T> LIBC_INLINE_VAR constexpr in_place_type_t<T> in_place_type{};

template <size_t I> struct in_place_index_t {
  explicit in_place_index_t() = default;
};
template <size_t I>
LIBC_INLINE_VAR constexpr in_place_index_t<I> in_place_index{};

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_UTILITY_IN_PLACE_H
