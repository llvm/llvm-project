//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// views::iota

#include <ranges>
#include <cassert>
#include <concepts>

#include "test_macros.h"
#include "types.h"

template<class T, class U>
constexpr void testType(U u) {
  // Test that this generally does the right thing.
  // Test with only one argument.
  {
    assert(*std::views::iota(T(0)).begin() == T(0));
  }
  {
    const auto io = std::views::iota(T(10));
    assert(*io.begin() == T(10));
  }
  // Test with two arguments.
  {
    assert(*std::views::iota(T(0), u).begin() == T(0));
  }
  {
    const auto io = std::views::iota(T(10), u);
    assert(*io.begin() == T(10));
  }
  // Test that we return the correct type.
  {
    ASSERT_SAME_TYPE(decltype(std::views::iota(T(10))), std::ranges::iota_view<T>);
    ASSERT_SAME_TYPE(decltype(std::views::iota(T(10), u)), std::ranges::iota_view<T, U>);
  }
}

struct X {};

constexpr bool test() {
  testType<SomeInt>(SomeInt(10));
  testType<SomeInt>(IntComparableWith(SomeInt(10)));
  testType<signed long>(IntComparableWith<signed long>(10));
  testType<unsigned long>(IntComparableWith<unsigned long>(10));
  testType<int>(IntComparableWith<int>(10));
  testType<int>(int(10));
  testType<unsigned>(unsigned(10));
  testType<unsigned>(IntComparableWith<unsigned>(10));
  testType<short>(short(10));
  testType<short>(IntComparableWith<short>(10));
  testType<unsigned short>(IntComparableWith<unsigned short>(10));

  {
    static_assert( std::is_invocable_v<decltype(std::views::iota), int>);
    static_assert(!std::is_invocable_v<decltype(std::views::iota), X>);
    static_assert( std::is_invocable_v<decltype(std::views::iota), int, int>);
    static_assert(!std::is_invocable_v<decltype(std::views::iota), int, X>);
  }
  {
    static_assert(std::same_as<decltype(std::views::iota), decltype(std::ranges::views::iota)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
