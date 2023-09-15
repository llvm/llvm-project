//===-- Holds an expected or unexpected value -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SUPPORT_CPP_EXPECTED_H
#define LLVM_LIBC_SUPPORT_CPP_EXPECTED_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/type_traits/always_false.h"
#include "src/__support/CPP/utility.h"

// BEWARE : This implementation is not fully conformant as it doesn't take
// `cpp::reference_wrapper` into account.
// It also doesn't currently implement constructors with 'cpp::in_place_t' and
// 'cpp::unexpect_t'. As a consequence, use of 'cpp::expected<T, E>' where 'T'
// and 'E' are the same type is likely to fail.

namespace __llvm_libc::cpp {

// This is used to hold an unexpected value so that a different constructor is
// selected.
template <class E> class unexpected {
  E err;

public:
  constexpr unexpected(const unexpected &) = default;
  constexpr unexpected(unexpected &&) = default;

  constexpr explicit unexpected(const E &e) : err(e) {}
  constexpr explicit unexpected(E &&e) : err(cpp::move(e)) {}

  constexpr const E &error() const & { return err; }
  constexpr E &error() & { return err; }
  constexpr const E &&error() const && { return cpp::move(err); }
  constexpr E &&error() && { return cpp::move(err); }
};

template <class E>
constexpr bool operator==(unexpected<E> &x, unexpected<E> &y) {
  return x.error() == y.error();
}
template <class E>
constexpr bool operator!=(unexpected<E> &x, unexpected<E> &y) {
  return !(x == y);
}

namespace detail {

// Returns exp.value() or cpp::move(exp.value()) if exp is an rvalue reference.
// It also retuns void iff E is an instance of cpp::expected<void, E>.
template <typename E> auto value(E &&exp) {
  using value_type = typename cpp::remove_cvref_t<E>::value_type;
  if constexpr (cpp::is_void_v<value_type>)
    return;
  else if constexpr (cpp::is_rvalue_reference_v<decltype(exp)>)
    return cpp::move(exp.value());
  else
    return exp.value();
}

// Returns exp.error() or cpp::move(exp.error()) if exp is an rvalue reference.
template <typename E> auto error(E &&exp) {
  if constexpr (cpp::is_rvalue_reference_v<decltype(exp)>)
    return cpp::move(exp.error());
  else
    return exp.error();
}

// Returns the function return type for unary functions.
template <typename T, class F>
struct function_return_type
    : cpp::type_identity<cpp::remove_cvref_t<cpp::invoke_result_t<F, T>>> {};

// Returns the function return type for nullary functions.
template <class F>
struct function_return_type<void, F>
    : cpp::type_identity<cpp::remove_cvref_t<cpp::invoke_result_t<F>>> {};

// Helper type.
template <typename T, class F>
using function_return_type_t = typename function_return_type<T, F>::type;

// Calls 'f' either directly when E = cpp::expected<void, ...> or with value()
// otherwise.
template <typename E, class F> decltype(auto) invoke(E &&exp, F &&f) {
  using T = typename cpp::remove_cvref_t<E>::value_type;
  if constexpr (is_void_v<T>)
    return cpp::invoke(cpp::forward<F>(f));
  else
    return cpp::invoke(cpp::forward<F>(f), value(cpp::forward<E>(exp)));
}

// We implement all monadic calls in a single 'apply' function below.
// This enum helps selecting the appropriate logic.
enum class Impl { TRANSFORM, TRANSFORM_ERROR, AND_THEN, OR_ELSE };

// Handles all flavors of 'transform', 'transform_error', 'and_then' and
// 'or_else':
// - 'exp' can be const or non-const.
// - 'exp' can be an 'lvalue_reference' or an 'rvalue_reference'.
//    Note: if 'exp' is an 'rvalue_reference' its value / error is moved.
// - 'f' can take zero or one argument.
// - 'f' can return void or another type.
template <Impl impl, typename E, class F> decltype(auto) apply(E &&exp, F &&f) {
  using value_type = decltype(value(cpp::forward<E>(exp)));
  using UnqualE = cpp::remove_cvref_t<E>;
  using T = typename UnqualE::value_type;
  if constexpr (impl == Impl::TRANSFORM || impl == Impl::AND_THEN) {
    // processing value
    using U = function_return_type_t<value_type, F>;
    using unexpected_u = typename UnqualE::unexpected_type;
    if constexpr (impl == Impl::TRANSFORM) {
      static_assert(!cpp::is_reference_v<U>,
                    "F should return a non-reference type");
      using expected_u = typename UnqualE::template rebind<U>;
      if (exp.has_value()) {
        if constexpr (is_void_v<U>) {
          invoke(cpp::forward<E>(exp), cpp::forward<F>(f));
          return expected_u();
        } else {
          return expected_u(invoke(cpp::forward<E>(exp), cpp::forward<F>(f)));
        }
      } else {
        return expected_u(unexpected_u(error(cpp::forward<E>(exp))));
      }
    }
    if constexpr (impl == Impl::AND_THEN) {
      if (exp.has_value())
        return invoke(cpp::forward<E>(exp), cpp::forward<F>(f));
      else
        return U(unexpected_u(error(cpp::forward<E>(exp))));
    }
  } else if constexpr (impl == Impl::OR_ELSE || impl == Impl::TRANSFORM_ERROR) {
    // processing error
    using error_type = decltype(error(cpp::forward<E>(exp)));
    using G = function_return_type_t<error_type, F>;
    if constexpr (impl == Impl::TRANSFORM_ERROR) {
      static_assert(!cpp::is_reference_v<G>,
                    "F should return a non-reference type");
      using expected_g = typename UnqualE::template rebind_error<G>;
      using unexpected_g = typename expected_g::unexpected_type;
      if (exp.has_value())
        if constexpr (is_void_v<T>)
          return expected_g();
        else
          return expected_g(value(cpp::forward<E>(exp)));
      else
        return expected_g(unexpected_g(
            cpp::invoke(cpp::forward<F>(f), error(cpp::forward<E>(exp)))));
    }
    if constexpr (impl == Impl::OR_ELSE) {
      if (exp.has_value()) {
        if constexpr (cpp::is_void_v<T>)
          return G();
        else
          return G(value(cpp::forward<E>(exp)));
      } else {
        return cpp::invoke(cpp::forward<F>(f), error(cpp::forward<E>(exp)));
      }
    }
  } else {
    static_assert(always_false<E>, "");
  }
}

template <typename E, class F> decltype(auto) transform(E &&exp, F &&f) {
  return apply<Impl::TRANSFORM>(cpp::forward<E>(exp), cpp::forward<F>(f));
}

template <typename E, class F> decltype(auto) transform_error(E &&exp, F &&f) {
  return apply<Impl::TRANSFORM_ERROR>(cpp::forward<E>(exp), cpp::forward<F>(f));
}

template <typename E, class F> decltype(auto) and_then(E &&exp, F &&f) {
  return apply<Impl::AND_THEN>(cpp::forward<E>(exp), cpp::forward<F>(f));
}

template <typename E, class F> decltype(auto) or_else(E &&exp, F &&f) {
  return apply<Impl::OR_ELSE>(cpp::forward<E>(exp), cpp::forward<F>(f));
}

} // namespace detail

template <class T, class E> class expected {
  static_assert(cpp::is_trivially_destructible_v<T> &&
                    cpp::is_trivially_destructible_v<E>,
                "prevents dealing with deletion of the union");

  union {
    T val;
    E err;
  };
  bool has_val;

public:
  using value_type = T;
  using error_type = E;
  using unexpected_type = cpp::unexpected<E>;
  template <class U> using rebind = cpp::expected<U, error_type>;
  template <class U> using rebind_error = cpp::expected<value_type, U>;

  constexpr expected() : expected(T{}) {}
  constexpr expected(T val) : val(val), has_val(true) {}
  constexpr expected(const expected &other) { *this = other; }
  constexpr expected(expected &&other) { *this = cpp::move(other); }
  constexpr expected(const unexpected<E> &err)
      : err(err.error()), has_val(false) {}
  constexpr expected(unexpected<E> &&err)
      : err(cpp::move(err.error())), has_val(false) {}

  constexpr expected &operator=(const expected &other) {
    if (other.has_value()) {
      val = other.value();
      has_val = true;
    } else {
      err = other.error();
      has_val = false;
    }
    return *this;
  }
  constexpr expected &operator=(expected &&other) {
    if (other.has_value()) {
      val = cpp::move(other.value());
      has_val = true;
    } else {
      err = cpp::move(other.error());
      has_val = false;
    }
    return *this;
  }
  constexpr expected &operator=(const unexpected<E> &other) {
    has_val = false;
    err = other.error();
  }
  constexpr expected &operator=(unexpected<E> &&other) {
    has_val = false;
    err = cpp::move(other.error());
  }

  constexpr bool has_value() const { return has_val; }
  constexpr operator bool() const { return has_val; }

  constexpr T &value() & { return val; }
  constexpr const T &value() const & { return val; }
  constexpr T &&value() && { return cpp::move(val); }
  constexpr const T &&value() const && { return cpp::move(val); }

  constexpr const E &error() const & { return err; }
  constexpr E &error() & { return err; }
  constexpr const E &&error() const && { return cpp::move(err); }
  constexpr E &&error() && { return cpp::move(err); }

  constexpr T *operator->() { return &val; }
  constexpr const T &operator*() const & { return val; }
  constexpr T &operator*() & { return val; }
  constexpr const T &&operator*() const && { return cpp::move(val); }
  constexpr T &&operator*() && { return cpp::move(val); }

  // value_or
  template <class U> constexpr T value_or(U &&default_value) const & {
    return has_value() ? **this
                       : static_cast<T>(cpp::forward<U>(default_value));
  }
  template <class U> constexpr T value_or(U &&default_value) && {
    return has_value() ? cpp::move(**this)
                       : static_cast<T>(cpp::forward<U>(default_value));
  }

  // transform
  template <class F> constexpr decltype(auto) transform(F &&f) & {
    return detail::transform(*this, f);
  }
  template <class F> constexpr decltype(auto) transform(F &&f) const & {
    return detail::transform(*this, f);
  }
  template <class F> constexpr decltype(auto) transform(F &&f) && {
    return detail::transform(*this, f);
  }
  template <class F> constexpr decltype(auto) transform(F &&f) const && {
    return detail::transform(*this, f);
  }

  // transform_error
  template <class F> constexpr decltype(auto) transform_error(F &&f) & {
    return detail::transform_error(*this, f);
  }
  template <class F> constexpr decltype(auto) transform_error(F &&f) const & {
    return detail::transform_error(*this, f);
  }
  template <class F> constexpr decltype(auto) transform_error(F &&f) && {
    return detail::transform_error(*this, f);
  }
  template <class F> constexpr decltype(auto) transform_error(F &&f) const && {
    return detail::transform_error(*this, f);
  }

  // and_then
  template <class F> constexpr decltype(auto) and_then(F &&f) & {
    return detail::and_then(*this, f);
  }
  template <class F> constexpr decltype(auto) and_then(F &&f) const & {
    return detail::and_then(*this, f);
  }
  template <class F> constexpr decltype(auto) and_then(F &&f) && {
    return detail::and_then(*this, f);
  }
  template <class F> constexpr decltype(auto) and_then(F &&f) const && {
    return detail::and_then(*this, f);
  }

  // or_else
  template <class F> constexpr decltype(auto) or_else(F &&f) & {
    return detail::or_else(*this, f);
  }
  template <class F> constexpr decltype(auto) or_else(F &&f) const & {
    return detail::or_else(*this, f);
  }
  template <class F> constexpr decltype(auto) or_else(F &&f) && {
    return detail::or_else(*this, f);
  }
  template <class F> constexpr decltype(auto) or_else(F &&f) const && {
    return detail::or_else(*this, f);
  }
};

template <class E> class expected<void, E> {
  static_assert(cpp::is_trivially_destructible_v<E>);

  E err;
  bool has_val;

public:
  using value_type = void;
  using error_type = E;
  using unexpected_type = cpp::unexpected<E>;
  template <class U> using rebind = cpp::expected<U, error_type>;
  template <class U> using rebind_error = cpp::expected<value_type, U>;

  constexpr expected() : has_val(true) {}
  constexpr expected(const expected &other) { *this = other; }
  constexpr expected(expected &&other) { *this = cpp::move(other); }
  constexpr expected(const unexpected<E> &err)
      : err(err.error()), has_val(false) {}
  constexpr expected(unexpected<E> &&err)
      : err(cpp::move(err.error())), has_val(false) {}

  constexpr expected &operator=(const expected &other) {
    if (other.has_value()) {
      has_val = true;
    } else {
      err = other.error();
      has_val = false;
    }
    return *this;
  }
  constexpr expected &operator=(expected &&other) {
    if (other.has_value()) {
      has_val = true;
    } else {
      err = cpp::move(other.error());
      has_val = false;
    }
    return *this;
  }
  constexpr expected &operator=(const unexpected<E> &other) {
    has_val = false;
    err = other.error();
  }
  constexpr expected &operator=(unexpected<E> &&other) {
    has_val = false;
    err = cpp::move(other.error());
  }

  constexpr bool has_value() const { return has_val; }
  constexpr operator bool() const { return has_val; }

  constexpr void value() const & {}
  constexpr void value() && {}

  constexpr const E &error() const & { return err; }
  constexpr E &error() & { return err; }
  constexpr const E &&error() const && { return cpp::move(err); }
  constexpr E &&error() && { return cpp::move(err); }

  constexpr void operator*() const {}

  // transform
  template <class F> constexpr decltype(auto) transform(F &&f) & {
    return detail::transform(*this, f);
  }
  template <class F> constexpr decltype(auto) transform(F &&f) const & {
    return detail::transform(*this, f);
  }
  template <class F> constexpr decltype(auto) transform(F &&f) && {
    return detail::transform(*this, f);
  }
  template <class F> constexpr decltype(auto) transform(F &&f) const && {
    return detail::transform(*this, f);
  }

  // transform_error
  template <class F> constexpr decltype(auto) transform_error(F &&f) & {
    return detail::transform_error(*this, f);
  }
  template <class F> constexpr decltype(auto) transform_error(F &&f) const & {
    return detail::transform_error(*this, f);
  }
  template <class F> constexpr decltype(auto) transform_error(F &&f) && {
    return detail::transform_error(*this, f);
  }
  template <class F> constexpr decltype(auto) transform_error(F &&f) const && {
    return detail::transform_error(*this, f);
  }

  // and_then
  template <class F> constexpr decltype(auto) and_then(F &&f) & {
    return detail::and_then(*this, f);
  }
  template <class F> constexpr decltype(auto) and_then(F &&f) const & {
    return detail::and_then(*this, f);
  }
  template <class F> constexpr decltype(auto) and_then(F &&f) && {
    return detail::and_then(*this, f);
  }
  template <class F> constexpr decltype(auto) and_then(F &&f) const && {
    return detail::and_then(*this, f);
  }

  // or_else
  template <class F> constexpr decltype(auto) or_else(F &&f) & {
    return detail::or_else(*this, f);
  }
  template <class F> constexpr decltype(auto) or_else(F &&f) const & {
    return detail::or_else(*this, f);
  }
  template <class F> constexpr decltype(auto) or_else(F &&f) && {
    return detail::or_else(*this, f);
  }
  template <class F> constexpr decltype(auto) or_else(F &&f) const && {
    return detail::or_else(*this, f);
  }
};

template <class T, class E>
constexpr bool operator==(const expected<T, E> &lhs,
                          const expected<T, E> &rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  if (lhs.has_value()) {
    if constexpr (cpp::is_void_v<T>)
      return true;
    else
      return *lhs == *rhs;
  }
  return lhs.error() == rhs.error();
}

template <class T, class E>
constexpr bool operator!=(const expected<T, E> &lhs,
                          const expected<T, E> &rhs) {
  return !(lhs == rhs);
}

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SUPPORT_CPP_EXPECTED_H
