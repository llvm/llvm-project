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
// template<class U> constexpr T value_or(U&& v) const &;
// Mandates: is_copy_constructible_v<T> is true and is_convertible_v<U, T> is true.

// template<class U> constexpr T value_or(U&& v) &&;
// Mandates: is_move_constructible_v<T> is true and is_convertible_v<U, T> is true.

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
    f1.value_or(5); // expected-note{{in instantiation of function template specialization 'std::expected<NonCopyable, int>::value_or<int>' requested here}}
    // expected-error-re@*:* {{static assertion failed {{.*}}value_type has to be copy constructible}}
  }

  // const & overload
  // !is_convertible_v<U, T>
  {
    const std::expected<NotConvertibleFromInt, int> f1{std::in_place};
    f1.value_or(5); // expected-note{{in instantiation of function template specialization 'std::expected<NotConvertibleFromInt, int>::value_or<int>' requested here}}
    //expected-error-re@*:* {{static assertion failed {{.*}}argument has to be convertible to value_type}}
  }

  // && overload
  // !is_move_constructible_v<T>,
  {
    std::expected<NonMovable, int> f1{5};
    std::move(f1).value_or(5); // expected-note{{in instantiation of function template specialization 'std::expected<NonMovable, int>::value_or<int>' requested here}}
    //expected-error-re@*:* {{static assertion failed {{.*}}value_type has to be move constructible}}
  }

  // && overload
  // !is_convertible_v<U, T>
  {
    std::expected<NotConvertibleFromInt, int> f1{std::in_place};
    std::move(f1).value_or(5); // expected-note{{in instantiation of function template specialization 'std::expected<NotConvertibleFromInt, int>::value_or<int>' requested here}}
    //expected-error-re@*:* {{static assertion failed {{.*}}argument has to be convertible to value_type}}
  }
}
