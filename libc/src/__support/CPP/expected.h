//===-- Holds an expected or unexpected value -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_EXPECTED_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_EXPECTED_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

// This is used to hold an unexpected value so that a different constructor is
// selected.
template <class T> class unexpected {
  T value;

public:
  LIBC_INLINE constexpr explicit unexpected(T value)
      : value(cpp::move(value)) {}

  LIBC_INLINE constexpr T &error() & { return value; }
  LIBC_INLINE constexpr const T &error() const & { return value; }
  LIBC_INLINE constexpr T &&error() && { return cpp::move(value); }
};

template <class T> explicit unexpected(T) -> unexpected<T>;

struct unexpect_t {
  LIBC_INLINE constexpr explicit unexpect_t() = default;
};

LIBC_INLINE_VAR constexpr unexpect_t unexpect{};

template <class T, class E,
          bool =
              is_trivially_destructible_v<T> && is_trivially_destructible_v<E>>
struct expected_storage;

// Trivial case: no destructor declared.
template <class T, class E> struct expected_storage<T, E, true> {
  union {
    T val;
    E err;
  };

  bool exp;

  template <typename... Args>
  LIBC_INLINE constexpr explicit expected_storage(in_place_t, Args &&...args)
      : val(cpp::forward<Args>(args)...), exp(true) {}

  template <typename... Args>
  LIBC_INLINE constexpr explicit expected_storage(unexpect_t, Args &&...args)
      : err(cpp::forward<Args>(args)...), exp(false) {}
};

// Non-trivial case: destructor destroys the active member.
template <class T, class E> struct expected_storage<T, E, false> {
  union {
    T val;
    E err;
  };

  bool exp;

  template <typename... Args>
  LIBC_INLINE constexpr explicit expected_storage(in_place_t, Args &&...args)
      : val(cpp::forward<Args>(args)...), exp(true) {}

  template <typename... Args>
  LIBC_INLINE constexpr explicit expected_storage(unexpect_t, Args &&...args)
      : err(cpp::forward<Args>(args)...), exp(false) {}

  LIBC_INLINE ~expected_storage() {
    if (exp)
      val.~T();
    else
      err.~E();
  }
};

template <class T, class E> class expected : private expected_storage<T, E> {
  using Base = expected_storage<T, E>;

public:
  LIBC_INLINE constexpr expected(const T &val) : Base(in_place, val) {}
  LIBC_INLINE constexpr expected(T &&val) : Base(in_place, cpp::move(val)) {}
  LIBC_INLINE constexpr expected(unexpected<E> unexp)
      : Base(unexpect, cpp::move(unexp).error()) {}

  LIBC_INLINE constexpr bool has_value() const { return this->exp; }

  LIBC_INLINE constexpr T &value() { return this->val; }
  LIBC_INLINE constexpr E &error() { return this->err; }
  LIBC_INLINE constexpr const T &value() const { return this->val; }
  LIBC_INLINE constexpr const E &error() const { return this->err; }

  LIBC_INLINE constexpr operator bool() const { return this->exp; }

  LIBC_INLINE constexpr T &operator*() { return this->val; }
  LIBC_INLINE constexpr const T &operator*() const { return this->val; }
  LIBC_INLINE constexpr T *operator->() { return &this->val; }
  LIBC_INLINE constexpr const T *operator->() const { return &this->val; }
};

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_EXPECTED_H
