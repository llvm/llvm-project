//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>
//
// template<class OtherElementType>
//   explicit constexpr aligned_accessor(
//     default_accessor<OtherElementType>) noexcept {}
//
// Constraints: is_convertible_v<OtherElementType(*)[], element_type(*)[]> is true.

#include <mdspan>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

struct Base {};
struct Derived : public Base {};

template <class FromT, class ToT, std::size_t ToN>
constexpr void test_conversion() {
  std::default_accessor<FromT> from;
  ASSERT_NOEXCEPT(std::aligned_accessor<ToT, ToN>(from));
  [[maybe_unused]] std::aligned_accessor<ToT, ToN> to(from);
  // check that the constructor is explicit
  static_assert(std::is_nothrow_constructible_v<std::aligned_accessor<ToT, ToN>, std::default_accessor<ToT>>);
  static_assert(!std::is_convertible_v<std::default_accessor<ToT>, std::aligned_accessor<ToT, ToN>>);
}

template <class From, class To>
constexpr void test_it() {
  constexpr std::size_t N = alignof(To);
  test_conversion<From, To, N>();
  test_conversion<From, To, 2 * N>();
  test_conversion<From, To, 4 * N>();
  test_conversion<From, To, 8 * N>();
  test_conversion<From, To, 16 * N>();
  test_conversion<From, To, 32 * N>();
}

constexpr bool test() {
  test_it<int, int>();
  test_it<int, const int>();
  test_it<const int, const int>();
  test_it<MinimalElementType, MinimalElementType>();
  test_it<MinimalElementType, const MinimalElementType>();
  test_it<const MinimalElementType, const MinimalElementType>();

  // char is convertible to int, but accessors are not
  static_assert(!std::is_constructible_v<std::aligned_accessor<int, 4>, std::default_accessor<char>>);
  // don't allow conversion from const elements to non-const
  static_assert(!std::is_constructible_v<std::aligned_accessor<int, 4>, std::default_accessor<const int>>);
  // MinimalElementType is constructible from int, but accessors should not be convertible
  static_assert(!std::is_constructible_v<std::aligned_accessor<MinimalElementType, 8>, std::default_accessor<int>>);
  // don't allow conversion from const elements to non-const
  static_assert(!std::is_constructible_v<std::aligned_accessor<MinimalElementType, 16>,
                                         std::default_accessor<const MinimalElementType>>);
  // don't allow conversion from Base to Derived
  static_assert(!std::is_constructible_v<std::aligned_accessor<Derived, 2>, std::default_accessor<Base>>);
  // don't allow conversion from Derived to Base
  static_assert(!std::is_constructible_v<std::aligned_accessor<Base, 2>, std::default_accessor<Derived>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
