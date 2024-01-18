//===-- A self contained equivalent of std::array ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_ARRAY_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_ARRAY_H

#include "src/__support/macros/attributes.h"
#include <stddef.h> // For size_t.

namespace LIBC_NAMESPACE {
namespace cpp {

template <class T, size_t N> struct array {
  static_assert(N != 0,
                "Cannot create a LIBC_NAMESPACE::cpp::array of size 0.");

  T Data[N];
  using value_type = T;
  using iterator = T *;
  using const_iterator = const T *;

  LIBC_INLINE constexpr T *data() { return Data; }
  LIBC_INLINE constexpr const T *data() const { return Data; }

  LIBC_INLINE constexpr T &front() { return Data[0]; }
  LIBC_INLINE constexpr T &front() const { return Data[0]; }

  LIBC_INLINE constexpr T &back() { return Data[N - 1]; }
  LIBC_INLINE constexpr T &back() const { return Data[N - 1]; }

  LIBC_INLINE constexpr T &operator[](size_t Index) { return Data[Index]; }

  LIBC_INLINE constexpr const T &operator[](size_t Index) const {
    return Data[Index];
  }

  LIBC_INLINE constexpr size_t size() const { return N; }

  LIBC_INLINE constexpr bool empty() const { return N == 0; }

  LIBC_INLINE constexpr iterator begin() { return Data; }
  LIBC_INLINE constexpr const_iterator begin() const { return Data; }

  LIBC_INLINE constexpr iterator end() { return Data + N; }
  LIBC_INLINE const_iterator end() const { return Data + N; }
};

} // namespace cpp
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_ARRAY_H
