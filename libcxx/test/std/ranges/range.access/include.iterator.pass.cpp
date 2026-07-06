//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// [range.access.general]/1:
// In addition to being available via inclusion of the <ranges> header, the customization point objects in
// [range.access] are available when the header <iterator> is included.

#include <iterator>
#include <type_traits>

#include "test_macros.h"

template <class CPO, class... Args>
constexpr void test(CPO& o, Args&&... args) {
  static_assert(std::is_const_v<CPO>);
  static_assert(std::is_class_v<CPO>);
  static_assert(std::is_trivially_copyable_v<CPO>);
  static_assert(std::is_trivially_default_constructible_v<CPO>);

  auto p  = o;
  using T = decltype(p);
  (void)o(args...); // to make sure the CPO can actually be used

  // The type of a customization point object, ignoring cv-qualifiers, shall model semiregular.
  static_assert(std::semiregular<T>);

  // The type T of a customization point object, ignoring cv-qualifiers, shall model...
  static_assert(std::invocable<T&, Args...>);
  static_assert(std::invocable<const T&, Args...>);
  static_assert(std::invocable<T, Args...>);
  static_assert(std::invocable<const T, Args...>);
}

int a[10];

constexpr bool test() {
  test(std::ranges::begin, a);
  test(std::ranges::end, a);
  test(std::ranges::cbegin, a);
  test(std::ranges::cdata, a);
  test(std::ranges::cend, a);
  test(std::ranges::crbegin, a);
  test(std::ranges::crend, a);
  test(std::ranges::data, a);
  test(std::ranges::empty, a);
  test(std::ranges::rbegin, a);
  test(std::ranges::rend, a);
  test(std::ranges::size, a);
  test(std::ranges::ssize, a);

#if TEST_STD_VER >= 26
  // test(std::views::reserve_hint, a);
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
