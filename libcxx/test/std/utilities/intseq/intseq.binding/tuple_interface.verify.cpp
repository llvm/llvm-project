//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <utility>

// template<size_t I, class T, T... Values>
//   struct tuple_element<I, integer_sequence<T, Values...>>;
// template<size_t I, class T, T... Values>
//   struct tuple_element<I, const integer_sequence<T, Values...>>;
// template<size_t I, class T, T... Values>
//   constexpr T get(integer_sequence<T, Values...>) noexcept;

// Expect failures for tuple_element and get with empty integer_sequence

#include <utility>

void test() {
  // expected-error-re@*:* {{static assertion failed{{.*}}Index out of bounds in std::tuple_element<> (std::integer_sequence)}}
  using test1 = std::tuple_element_t<0, std::integer_sequence<int>>;
  // expected-error-re@*:* {{static assertion failed{{.*}}Index out of bounds in std::tuple_element<> (const std::integer_sequence)}}
  using test2 = std::tuple_element_t<0, const std::integer_sequence<int>>;

  std::integer_sequence<int> empty;
  // expected-error-re@*:* {{static assertion failed{{.*}}Index out of bounds in std::get<> (std::integer_sequence)}}
  // expected-error-re@*:* {{invalid index 0 for pack {{.*}} of size 0}}
  (void)std::get<0>(empty);
}
