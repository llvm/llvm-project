//===-- Holds an expected or unexpected value -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_EXPECTED_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_EXPECTED_H

namespace LIBC_NAMESPACE::cpp {

// This is used to hold an unexpected value so that a different constructor is
// selected.
template <class T> class unexpected {
  T value;

public:
  constexpr explicit unexpected(T value) : value(value) {}
  constexpr T error() { return value; }
};

template <class T, class E> class expected {
  union {
    T exp;
    E unexp;
  };
  bool is_expected;

public:
  constexpr expected(T exp) : exp(exp), is_expected(true) {}
  constexpr expected(unexpected<E> unexp)
      : unexp(unexp.error()), is_expected(false) {}

  constexpr bool has_value() const { return is_expected; }

  constexpr T &value() { return exp; }
  constexpr E &error() { return unexp; }
  constexpr const T &value() const { return exp; }
  constexpr const E &error() const { return unexp; }

  constexpr operator bool() const { return is_expected; }

  constexpr T &operator*() { return exp; }
  constexpr const T &operator*() const { return exp; }
  constexpr T *operator->() { return &exp; }
  constexpr const T *operator->() const { return &exp; }
};

} // namespace LIBC_NAMESPACE::cpp

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_EXPECTED_H
