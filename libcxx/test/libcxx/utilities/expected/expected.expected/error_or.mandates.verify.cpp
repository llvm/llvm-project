//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test the mandates
// template<class G = E> constexpr E error_or(G&&) const &;
// Mandates: is_copy_constructible_v<G> is true and is_convertible_v<U, E> is true.

// template<class G = E> constexpr E error_or(G&&) &&;
// Mandates: is_move_constructible_v<G> is true and is_convertible_v<U, E> is true.

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

// clang-format off
void test() {
  // const & overload
  // !is_copy_constructible_v<G>,
  {
    const std::expected<int, NonCopyable> f1(std::unexpect, 0);
    f1.error_or(5); // expected-note{{in instantiation of function template specialization 'std::expected<int, NonCopyable>::error_or<int>' requested here}}
    // expected-error-re@*:* {{static assertion failed {{.*}}error_type has to be copy constructible}}
    // expected-error-re@*:* {{call to deleted constructor of{{.*}}}}
  }

  // const & overload
  // !is_convertible_v<U, T>
  {
    const std::expected<int, NotConvertibleFromInt> f1(std::unexpect, NotConvertibleFromInt{});
    f1.error_or(5); // expected-note{{in instantiation of function template specialization 'std::expected<int, NotConvertibleFromInt>::error_or<int>' requested here}}
    // expected-error-re@*:* {{static assertion failed {{.*}}argument has to be convertible to error_type}}
    // expected-error-re@*:* {{no viable conversion from returned value of type{{.*}}}}

  }

  // && overload
  // !is_move_constructible_v<T>,
  {
    std::expected<int, NonMovable> f1(std::unexpect, 0);
    std::move(f1).error_or(5); // expected-note{{in instantiation of function template specialization 'std::expected<int, NonMovable>::error_or<int>' requested here}}
    // expected-error-re@*:* {{static assertion failed {{.*}}error_type has to be move constructible}}
    // expected-error-re@*:* {{call to deleted constructor of{{.*}}}}
  }

  // && overload
  // !is_convertible_v<U, T>
  {
    std::expected<int, NotConvertibleFromInt> f1(std::unexpect, NotConvertibleFromInt{});
    std::move(f1).error_or(5); // expected-note{{in instantiation of function template specialization 'std::expected<int, NotConvertibleFromInt>::error_or<int>' requested here}}
    //expected-error-re@*:* {{static assertion failed {{.*}}argument has to be convertible to error_type}}
    // expected-error-re@*:* {{no viable conversion from returned value of type{{.*}}}}
  }
}
// clang-format on
