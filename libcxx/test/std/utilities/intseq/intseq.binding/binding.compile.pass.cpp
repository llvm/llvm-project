//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <utility>

// template<class T, T... Values>
//   struct tuple_size<integer_sequence<T, Values...>>;
// template<size_t I, class T, T... Values>
//   struct tuple_element<I, integer_sequence<T, Values...>>;
// template<size_t I, class T, T... Values>
//   struct tuple_element<I, const integer_sequence<T, Values...>>;
// template<size_t I, class T, T... Values>
//   constexpr T get(integer_sequence<T, Values...>) noexcept;

#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>

constexpr void test() {
  // std::tuple_size_v
  using empty = std::integer_sequence<int>;
  static_assert(std::tuple_size_v<empty> == 0);
  static_assert(std::tuple_size_v<const empty> == 0);

  using size4 = std::integer_sequence<int, 9, 8, 7, 2>;
  static_assert(std::tuple_size_v<size4> == 4);
  static_assert(std::tuple_size_v<const size4> == 4);

  // std::tuple_element_t
  static_assert(std::is_same_v<std::tuple_element_t<0, size4>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<1, size4>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<2, size4>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<3, size4>, int>);

  static_assert(std::is_same_v<std::tuple_element_t<0, const size4>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<1, const size4>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<2, const size4>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<3, const size4>, int>);

  // std::get
  constexpr static size4 seq4{};
  static_assert(get<0>(seq4) == 9);
  static_assert(get<1>(seq4) == 8);
  static_assert(get<2>(seq4) == 7);
  static_assert(get<3>(seq4) == 2);
}
