//===-- Standalone implementation of std::optional --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/macros/attributes.h"

namespace __llvm_libc {
namespace cpp {

// Trivial in_place_t struct.
struct in_place_t {
  LIBC_INLINE_VAR explicit in_place_t() = default;
};

// Trivial nullopt_t struct.
struct nullopt_t {
  LIBC_INLINE_VAR explicit nullopt_t() = default;
};

// nullopt that can be used and returned.
LIBC_INLINE_VAR constexpr nullopt_t nullopt{};

// in_place that can be used in the constructor.
LIBC_INLINE_VAR constexpr in_place_t in_place{};

// This is very simple implementation of the std::optional class. It makes
// several assumptions that the underlying type is trivially constructable,
// copyable, or movable.
template <typename T> class optional {
  template <typename U> class OptionalStorage {
    union {
      char empty;
      U stored_value;
    };
    bool in_use;

  public:
    LIBC_INLINE ~OptionalStorage() { reset(); }

    LIBC_INLINE constexpr OptionalStorage() : empty(), in_use(false) {}

    template <typename... Args>
    LIBC_INLINE constexpr explicit OptionalStorage(in_place_t, Args &&...args)
        : stored_value(forward<Args>(args)...), in_use(true) {}

    LIBC_INLINE void reset() {
      if (in_use)
        stored_value.~U();
      in_use = false;
    }

    LIBC_INLINE constexpr bool has_value() const { return in_use; }

    LIBC_INLINE U &value() & { return stored_value; }
    LIBC_INLINE constexpr U const &value() const & { return stored_value; }
    LIBC_INLINE U &&value() && { return move(stored_value); }
  };

  OptionalStorage<T> storage;

public:
  LIBC_INLINE constexpr optional() = default;
  LIBC_INLINE constexpr optional(nullopt_t) {}

  LIBC_INLINE constexpr optional(const T &t) : storage(in_place, t) {}
  LIBC_INLINE constexpr optional(const optional &) = default;

  LIBC_INLINE constexpr optional(T &&t) : storage(in_place, move(t)) {}
  LIBC_INLINE constexpr optional(optional &&O) = default;

  template <typename... ArgTypes>
  LIBC_INLINE constexpr optional(in_place_t, ArgTypes &&...Args)
      : storage(in_place, forward<ArgTypes>(Args)...) {}

  LIBC_INLINE optional &operator=(T &&t) {
    storage = move(t);
    return *this;
  }
  LIBC_INLINE optional &operator=(optional &&) = default;

  LIBC_INLINE static constexpr optional create(const T *t) {
    return t ? optional(*t) : optional();
  }

  LIBC_INLINE optional &operator=(const T &t) {
    storage = t;
    return *this;
  }
  LIBC_INLINE optional &operator=(const optional &) = default;

  LIBC_INLINE void reset() { storage.reset(); }

  LIBC_INLINE constexpr const T &value() const & { return storage.value(); }
  LIBC_INLINE T &value() & { return storage.value(); }

  LIBC_INLINE constexpr explicit operator bool() const { return has_value(); }
  LIBC_INLINE constexpr bool has_value() const { return storage.has_value(); }
  LIBC_INLINE constexpr const T *operator->() const { return &storage.value(); }
  LIBC_INLINE T *operator->() { return &storage.value(); }
  LIBC_INLINE constexpr const T &operator*() const & { return value(); }
  LIBC_INLINE T &operator*() & { return value(); }

  template <typename U> LIBC_INLINE constexpr T value_or(U &&value) const & {
    return has_value() ? value() : forward<U>(value);
  }

  LIBC_INLINE T &&value() && { return move(storage.value()); }
  LIBC_INLINE T &&operator*() && { return move(storage.value()); }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H
