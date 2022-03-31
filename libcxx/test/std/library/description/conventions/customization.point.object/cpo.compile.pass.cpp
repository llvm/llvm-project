//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// [customization.point.object]
// [range.adaptor.object] "A range adaptor object is a customization point object..."

#include <compare>
#include <concepts>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

// Test for basic properties of C++20 16.3.3.3.6 [customization.point.object].
template <class CPO, class... Args>
constexpr bool test(CPO& o, Args&&...) {
  static_assert(std::is_const_v<CPO>);
  static_assert(std::is_class_v<CPO>);
  static_assert(std::is_trivial_v<CPO>);

  auto p = o;
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
//int arrays[10][10];
//std::pair<int, int> pairs[10];

// [concept.swappable]
static_assert(test(std::ranges::swap, a, a));

// [iterator.cust]
static_assert(test(std::ranges::iter_move, a + 0));
static_assert(test(std::ranges::iter_swap, a + 0, a + 1));

// [cmp.alg]
static_assert(test(std::partial_order, 1, 2));
static_assert(test(std::strong_order, 1, 2));
static_assert(test(std::weak_order, 1, 2));
static_assert(test(std::compare_partial_order_fallback, 1, 2));
static_assert(test(std::compare_strong_order_fallback, 1, 2));
static_assert(test(std::compare_weak_order_fallback, 1, 2));

// [range.access]
static_assert(test(std::ranges::begin, a));
static_assert(test(std::ranges::end, a));
static_assert(test(std::ranges::cbegin, a));
static_assert(test(std::ranges::cdata, a));
static_assert(test(std::ranges::cend, a));
//static_assert(test(std::ranges::crbegin, a));
//static_assert(test(std::ranges::crend, a));
static_assert(test(std::ranges::data, a));
static_assert(test(std::ranges::empty, a));
//static_assert(test(std::ranges::rbegin, a));
//static_assert(test(std::ranges::rend, a));
static_assert(test(std::ranges::size, a));
static_assert(test(std::ranges::ssize, a));

// [range.factories]
// views::empty<T> is not a CPO
static_assert(test(std::views::iota, 1));
static_assert(test(std::views::iota, 1, 10));
//static_assert(test(std::views::istream<int>, 1);
//static_assert(test(std::views::single, 4));

// [range.adaptors]
static_assert(test(std::views::all, a));
static_assert(test(std::views::common, a));
static_assert(test(std::views::counted, a, 10));
//static_assert(test(std::views::drop, a, 10));
//static_assert(test(std::views::drop_while, a, [](int x){ return x < 10; }));
//static_assert(test(std::views::elements<0>, pairs));
//static_assert(test(std::views::filter, a, [](int x){ return x < 10; }));
//static_assert(test(std::views::join, arrays));
static_assert(test(std::views::reverse, a));
//static_assert(test(std::views::split, a, 4));
//static_assert(test(std::views::take, a, 10));
//static_assert(test(std::views::take_while, a, [](int x){ return x < 10; }));
static_assert(test(std::views::transform, a, [](int x){ return x + 1; }));
