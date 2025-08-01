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
constexpr bool test(CPO& o, Args&&...) {
  static_assert(std::is_const_v<CPO>);
  static_assert(std::is_class_v<CPO>);
  static_assert(std::is_trivially_copyable_v<CPO>);
  static_assert(std::is_trivially_default_constructible_v<CPO>);

  auto p  = o;
  using T = decltype(p);

  // The type of a customization point object, ignoring cv-qualifiers, shall model semiregular.
  static_assert(std::semiregular<T>);

  // The type T of a customization point object, ignoring cv-qualifiers, shall model...
  static_assert(std::invocable<T&, Args...>);
  static_assert(std::invocable<const T&, Args...>);
  static_assert(std::invocable<T, Args...>);
  static_assert(std::invocable<const T, Args...>);

  return true;
}

int a[10];

static_assert(test(std::ranges::begin, a));
static_assert(test(std::ranges::end, a));
static_assert(test(std::ranges::cbegin, a));
static_assert(test(std::ranges::cdata, a));
static_assert(test(std::ranges::cend, a));
static_assert(test(std::ranges::crbegin, a));
static_assert(test(std::ranges::crend, a));
static_assert(test(std::ranges::data, a));
static_assert(test(std::ranges::empty, a));
static_assert(test(std::ranges::rbegin, a));
static_assert(test(std::ranges::rend, a));
static_assert(test(std::ranges::size, a));
static_assert(test(std::ranges::ssize, a));

#if TEST_STD_VER >= 26
// static_assert(test(std::views::reserve_hint, a));
#endif
