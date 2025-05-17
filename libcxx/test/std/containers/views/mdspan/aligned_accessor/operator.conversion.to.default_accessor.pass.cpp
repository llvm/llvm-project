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
//   constexpr operator default_accessor<OtherElementType>() const noexcept;
//
// Constraints: is_convertible_v<element_type(*)[], OtherElementType(*)[]> is true.

#include <mdspan>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

struct Base {};
struct Derived : public Base {};

template <class FromT, size_t FromN, class ToT>
constexpr void test_conversion() {
  std::aligned_accessor<FromT, FromN> from;
  ASSERT_NOEXCEPT(std::default_accessor<ToT>(from));
  [[maybe_unused]] std::default_accessor<ToT> to(from);
  // check that the conversion is implicit
  static_assert(std::is_nothrow_convertible_v<std::aligned_accessor<FromT, FromN>, std::default_accessor<ToT>>);
  static_assert(std::is_nothrow_constructible_v<std::default_accessor<ToT>, std::aligned_accessor<FromT, FromN>>);
}

template <class From, class To>
constexpr void test_it() {
  constexpr size_t N = alignof(From);
  test_conversion<From, N, To>();
  test_conversion<From, 2 * N, To>();
  test_conversion<From, 4 * N, To>();
  test_conversion<From, 8 * N, To>();
  test_conversion<From, 16 * N, To>();
  test_conversion<From, 32 * N, To>();
}

constexpr bool test() {
  test_it<int, int>();
  test_it<int, const int>();
  test_it<const int, const int>();
  test_it<MinimalElementType, MinimalElementType>();
  test_it<MinimalElementType, const MinimalElementType>();
  test_it<const MinimalElementType, const MinimalElementType>();

  // char is convertible to int, but accessors are not
  static_assert(!std::is_constructible_v<std::default_accessor<int>, std::aligned_accessor<char, 4>>);
  // don't allow conversion from const elements to non-const
  static_assert(!std::is_constructible_v<std::default_accessor<int>, std::aligned_accessor<const int, 8>>);
  // MinimalElementType is constructible from int, but accessors should not be convertible
  static_assert(!std::is_constructible_v<std::default_accessor<MinimalElementType>, std::aligned_accessor<int, 4>>);
  // don't allow conversion from const elements to non-const
  static_assert(!std::is_constructible_v<std::default_accessor<MinimalElementType>,
                                         std::aligned_accessor<const MinimalElementType, 4>>);
  // don't allow conversion from Base to Derived
  static_assert(!std::is_constructible_v<std::default_accessor<Derived>, std::aligned_accessor<Base, 1>>);
  // don't allow conversion from Derived to Base
  static_assert(!std::is_constructible_v<std::default_accessor<Base>, std::aligned_accessor<Derived, 1>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
