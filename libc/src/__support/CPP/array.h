//===-- A self contained equivalent of std::array ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_ARRAY_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_ARRAY_H

#include <stddef.h> // For size_t.

namespace __llvm_libc {
namespace cpp {

template <class T, size_t N> struct array {
  static_assert(N != 0, "Cannot create a __llvm_libc::cpp::array of size 0.");

  T Data[N];
  using value_type = T;
  using iterator = T *;
  using const_iterator = const T *;

  constexpr T *data() { return Data; }
  constexpr const T *data() const { return Data; }

  constexpr T &front() { return Data[0]; }
  constexpr T &front() const { return Data[0]; }

  constexpr T &back() { return Data[N - 1]; }
  constexpr T &back() const { return Data[N - 1]; }

  constexpr T &operator[](size_t Index) { return Data[Index]; }

  constexpr const T &operator[](size_t Index) const { return Data[Index]; }

  constexpr size_t size() const { return N; }

  constexpr bool empty() const { return N == 0; }

  constexpr iterator begin() { return Data; }
  constexpr const_iterator begin() const { return Data; }

  constexpr iterator end() { return Data + N; }
  const_iterator end() const { return Data + N; }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_ARRAY_H
