//===-- include/flang/Common/optional.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of std::optional borrowed from LLVM's
// libc/src/__support/CPP/optional.h with modifications (e.g. value_or, emplace
// methods were added).
//
// The implementation defines optional in Fortran::common namespace.
// This standalone implementation may be used if the target
// does not support std::optional implementation (e.g. CUDA device env),
// otherwise, Fortran::common::optional is an alias for std::optional.
//
// TODO: using libcu++ is the best option for CUDA, but there is a couple
// of issues:
//   * Older CUDA toolkits' libcu++ implementations do not support optional.
//   * The include paths need to be set up such that all STD header files
//     are taken from libcu++.
//   * cuda:: namespace need to be forced for all std:: references.
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_COMMON_OPTIONAL_H
#define FORTRAN_COMMON_OPTIONAL_H

#include "api-attrs.h"
#include <optional>

#if !defined(STD_OPTIONAL_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_OPTIONAL_UNSUPPORTED 1
#endif

#define FORTRAN_OPTIONAL_INLINE_WITH_ATTRS inline RT_API_ATTRS
#define FORTRAN_OPTIONAL_INLINE inline
#define FORTRAN_OPTIONAL_INLINE_VAR inline

namespace Fortran::common {

#if STD_OPTIONAL_UNSUPPORTED
// Trivial nullopt_t struct.
struct nullopt_t {
  constexpr explicit nullopt_t() = default;
};

// nullopt that can be used and returned.
FORTRAN_OPTIONAL_INLINE_VAR constexpr nullopt_t nullopt{};

// This is very simple implementation of the std::optional class. It makes
// several assumptions that the underlying type is trivially constructible,
// copyable, or movable.
template <typename T> class optional {
  template <typename U, bool = !std::is_trivially_destructible<U>::value>
  struct OptionalStorage {
    union {
      char empty;
      U stored_value;
    };

    bool in_use = false;

    FORTRAN_OPTIONAL_INLINE_WITH_ATTRS ~OptionalStorage() { reset(); }

    FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr OptionalStorage() : empty() {}

    template <typename... Args>
    FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr explicit OptionalStorage(
        std::in_place_t, Args &&...args)
        : stored_value(std::forward<Args>(args)...) {}

    FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr void reset() {
      if (in_use)
        stored_value.~U();
      in_use = false;
    }
  };

  // The only difference is that this type U doesn't have a nontrivial
  // destructor.
  template <typename U> struct OptionalStorage<U, false> {
    union {
      char empty;
      U stored_value;
    };

    bool in_use = false;

    FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr OptionalStorage() : empty() {}

    template <typename... Args>
    FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr explicit OptionalStorage(
        std::in_place_t, Args &&...args)
        : stored_value(std::forward<Args>(args)...) {}

    FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr void reset() {
      in_use = false;
    }
  };

  OptionalStorage<T> storage;

public:
  // The default methods do not use RT_API_ATTRS, which causes
  // warnings in CUDA compilation of form:
  //   __device__/__host__ annotation is ignored on a function .* that is
  //   explicitly defaulted on its first declaration
  FORTRAN_OPTIONAL_INLINE constexpr optional() = default;
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr optional(nullopt_t) {}

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr optional(const T &t)
      : storage(std::in_place, t) {
    storage.in_use = true;
  }
  FORTRAN_OPTIONAL_INLINE constexpr optional(const optional &) = default;

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr optional(T &&t)
      : storage(std::in_place, std::move(t)) {
    storage.in_use = true;
  }
  FORTRAN_OPTIONAL_INLINE constexpr optional(optional &&O) = default;

  template <typename... ArgTypes>
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr optional(
      std::in_place_t, ArgTypes &&...Args)
      : storage(std::in_place, std::forward<ArgTypes>(Args)...) {
    storage.in_use = true;
  }

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr optional &operator=(T &&t) {
    storage.stored_value = std::move(t);
    storage.in_use = true;
    return *this;
  }

  FORTRAN_OPTIONAL_INLINE constexpr optional &operator=(optional &&) = default;

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr optional &operator=(const T &t) {
    storage.stored_value = t;
    storage.in_use = true;
    return *this;
  }

  FORTRAN_OPTIONAL_INLINE constexpr optional &operator=(
      const optional &) = default;

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr void reset() { storage.reset(); }

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr const T &value() const & {
    return storage.stored_value;
  }

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr T &value() & {
    return storage.stored_value;
  }

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr explicit operator bool() const {
    return storage.in_use;
  }
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr bool has_value() const {
    return storage.in_use;
  }
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr const T *operator->() const {
    return &storage.stored_value;
  }
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr T *operator->() {
    return &storage.stored_value;
  }
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr const T &operator*() const & {
    return storage.stored_value;
  }
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr T &operator*() & {
    return storage.stored_value;
  }

  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr T &&value() && {
    return std::move(storage.stored_value);
  }
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr T &&operator*() && {
    return std::move(storage.stored_value);
  }

  template <typename VT>
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr T value_or(
      VT &&default_value) const & {
    return storage.in_use ? storage.stored_value
                          : static_cast<T>(std::forward<VT>(default_value));
  }

  template <typename VT>
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr T value_or(
      VT &&default_value) && {
    return storage.in_use ? std::move(storage.stored_value)
                          : static_cast<T>(std::forward<VT>(default_value));
  }

  template <typename... ArgTypes>
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS
      std::enable_if_t<std::is_constructible_v<T, ArgTypes &&...>, T &>
      emplace(ArgTypes &&...args) {
    reset();
    new (reinterpret_cast<void *>(std::addressof(storage.stored_value)))
        T(std::forward<ArgTypes>(args)...);
    storage.in_use = true;
    return value();
  }

  template <typename U = T,
      std::enable_if_t<(std::is_constructible_v<T, U &&> &&
                           !std::is_same_v<std::decay_t<U>, std::in_place_t> &&
                           !std::is_same_v<std::decay_t<U>, optional> &&
                           std::is_convertible_v<U &&, T>),
          bool> = true>
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS constexpr optional(U &&value) {
    new (reinterpret_cast<void *>(std::addressof(storage.stored_value)))
        T(std::forward<U>(value));
    storage.in_use = true;
  }

  template <typename U = T,
      std::enable_if_t<(std::is_constructible_v<T, U &&> &&
                           !std::is_same_v<std::decay_t<U>, std::in_place_t> &&
                           !std::is_same_v<std::decay_t<U>, optional> &&
                           !std::is_convertible_v<U &&, T>),
          bool> = false>
  FORTRAN_OPTIONAL_INLINE_WITH_ATTRS explicit constexpr optional(U &&value) {
    new (reinterpret_cast<void *>(std::addressof(storage.stored_value)))
        T(std::forward<U>(value));
    storage.in_use = true;
  }
};
#else // !STD_OPTIONAL_UNSUPPORTED
using std::nullopt;
using std::nullopt_t;
using std::optional;
#endif // !STD_OPTIONAL_UNSUPPORTED

// Leaves an optional disengaged, but with its payload storage written rather
// than left indeterminate.
//
// Predicates of the form `opt && x < *opt` test the engaged flag before
// reading the payload, so they never use an indeterminate value.  Compilers
// do, however, routinely if-convert the short-circuit `&&` into a branchless
// compare and select, which speculates the payload load.  On targets where
// that happens the select's condition is computed from indeterminate storage,
// and memory checkers then report a conditional branch that depends on
// uninitialized memory.  This matters for runtime objects that are
// placement-new'd into malloc'd storage, where a checker can see that the
// payload bytes were never written.
//
// Writing the payload once at construction makes those bytes defined without
// changing any observable behavior: the optional is still disengaged, and the
// speculated read is still meaningless.  The assignment below is therefore NOT
// dead code -- it exists solely for its effect on the payload storage, and
// reset() on a trivially destructible payload merely clears the engaged flag.
template <typename A>
FORTRAN_OPTIONAL_INLINE_WITH_ATTRS void ResetWithDefinedPayload(
    optional<A> &opt) {
  opt = A{};
  opt.reset();
}

} // namespace Fortran::common

#endif // FORTRAN_COMMON_OPTIONAL_H
