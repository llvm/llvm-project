//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>
//
// Test converting constructor:
//
// template<class OtherElementType>
//   constexpr default_accessor(default_accessor<OtherElementType>) noexcept {}
//
// Constraints: is_convertible_v<OtherElementType(*)[], element_type(*)[]> is true.

#include <mdspan>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

struct Base {};
struct Derived: public Base {};

template <class FromT, class ToT>
constexpr void test_conversion() {
  std::default_accessor<FromT> acc_from;
  ASSERT_NOEXCEPT(std::default_accessor<ToT>(acc_from));
  [[maybe_unused]] std::default_accessor<ToT> acc_to(acc_from);
}

constexpr bool test() {
  // default accessor conversion largely behaves like pointer conversion
  test_conversion<int, int>();
  test_conversion<int, const int>();
  test_conversion<const int, const int>();
  test_conversion<MinimalElementType, MinimalElementType>();
  test_conversion<MinimalElementType, const MinimalElementType>();
  test_conversion<const MinimalElementType, const MinimalElementType>();

  // char is convertible to int, but accessors are not
  static_assert(!std::is_constructible_v<std::default_accessor<int>, std::default_accessor<char>>);
  // don't allow conversion from const elements to non-const
  static_assert(!std::is_constructible_v<std::default_accessor<int>, std::default_accessor<const int>>);
  // MinimalElementType is constructible from int, but accessors should not be convertible
  static_assert(!std::is_constructible_v<std::default_accessor<MinimalElementType>, std::default_accessor<int>>);
  // don't allow conversion from const elements to non-const
  static_assert(!std::is_constructible_v<std::default_accessor<MinimalElementType>, std::default_accessor<const MinimalElementType>>);
  // don't allow conversion from Base to Derived
  static_assert(!std::is_constructible_v<std::default_accessor<Derived>, std::default_accessor<Base>>);
  // don't allow conversion from Derived to Base
  static_assert(!std::is_constructible_v<std::default_accessor<Base>, std::default_accessor<Derived>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
