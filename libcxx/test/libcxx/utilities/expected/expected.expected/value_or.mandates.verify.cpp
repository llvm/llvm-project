//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

// Test the mandates
// template<class U> constexpr remove_cv_t<T> value_or(U&& v) const &;
// Mandates: is_convertible_v<const T&, remove_cv_t<T>> is true and is_convertible_v<U, remove_cv_t<T>> is true.

// template<class U> constexpr remove_cv_t<T> value_or(U&& v) &&;
// Mandates: is_convertible_v<T, remove_cv_t<T>> is true and is_convertible_v<U, remove_cv_t<T>> is true.

#include <expected>
#include <utility>

struct NonCopyable {
  NonCopyable(int) {}
  NonCopyable(const NonCopyable&) = delete;
};

struct NonMovable {
  NonMovable(int) {}
  NonMovable(NonMovable&&) = delete;
};

struct NotConvertibleFromInt {};

void test() {
  // const & overload
  // !is_copy_constructible_v<T>,
  {
    const std::expected<NonCopyable, int> f1{5};
    // expected-note@+1 {{in instantiation of function template specialization 'std::expected<NonCopyable, int>::value_or<int>' requested here}}
    (void)f1.value_or(5);
    // expected-error-re@*:* {{static assertion failed {{.*}}must be implicitly convertible to remove_cv_t<T>}}
  }

  // const & overload
  // !is_convertible_v<U, T>
  {
    const std::expected<NotConvertibleFromInt, int> f1{std::in_place};
    // expected-note@+1 {{in instantiation of function template specialization 'std::expected<NotConvertibleFromInt, int>::value_or<int>' requested here}}
    (void)f1.value_or(5);
    //expected-error-re@*:* {{static assertion failed {{.*}}U must be implicitly convertible to remove_cv_t<T>}}
  }

  // && overload
  // !is_move_constructible_v<T>,
  {
    std::expected<NonMovable, int> f1{5};
    // expected-note@+1 {{in instantiation of function template specialization 'std::expected<NonMovable, int>::value_or<int>' requested here}}
    (void)std::move(f1).value_or(5);
    //expected-error-re@*:* {{static assertion failed {{.*}}must be implicitly convertible to remove_cv_t<T>}}
  }

  // && overload
  // !is_convertible_v<U, T>
  {
    std::expected<NotConvertibleFromInt, int> f1{std::in_place};
    // expected-note@+1 {{in instantiation of function template specialization 'std::expected<NotConvertibleFromInt, int>::value_or<int>' requested here}}
    (void)std::move(f1).value_or(5);
    //expected-error-re@*:* {{static assertion failed {{.*}}U must be implicitly convertible to remove_cv_t<T>}}
  }
}
