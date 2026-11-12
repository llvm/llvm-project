//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <class T>
// views::repeat(T &&) requires constructible_from<ranges::repeat_view<T>, T>;

// template <class T, class Bound>
// views::repeat(T &&, Bound &&) requires constructible_from<ranges::repeat_view<T, Bound>, T, Bound>;

#include <cassert>
#include <concepts>
#include <ranges>
#include <tuple>
#include <type_traits>

#include "MoveOnly.h"

struct NonCopyable {
  NonCopyable(NonCopyable&) = delete;
};

struct NonDefaultCtor {
  NonDefaultCtor(int) {}
};

struct Empty {};

struct LessThan3 {
  constexpr bool operator()(int i) const { return i < 3; }
};

struct EqualTo33 {
  constexpr bool operator()(int i) const { return i == 33; }
};

struct Add3 {
  constexpr int operator()(int i) const { return i + 3; }
};

// Tp is_object
static_assert(std::is_invocable_v<decltype(std::views::repeat), int>);
static_assert(!std::is_invocable_v<decltype(std::views::repeat), void>);

// _Bound is semiregular, integer like or std::unreachable_sentinel_t
static_assert(!std::is_invocable_v<decltype(std::views::repeat), int, Empty>);
static_assert(!std::is_invocable_v<decltype(std::views::repeat), int, NonCopyable>);
static_assert(!std::is_invocable_v<decltype(std::views::repeat), int, NonDefaultCtor>);
static_assert(std::is_invocable_v<decltype(std::views::repeat), int, std::unreachable_sentinel_t>);

// Tp is copy_constructible
static_assert(!std::is_invocable_v<decltype(std::views::repeat), NonCopyable>);

// Tp is move_constructible
static_assert(std::is_invocable_v<decltype(std::views::repeat), MoveOnly>);

// Test LWG4054 "Repeating a repeat_view should repeat the view"
static_assert(std::is_same_v<decltype(std::views::repeat(std::views::repeat(42))),
                             std::ranges::repeat_view<std::ranges::repeat_view<int>>>);

// These cases are from LWG4053, but they are actually covered by the resolution of LWG4054,
// and the resolution of LWG4053 only affects CTAD.
using RPV = std::ranges::repeat_view<const char*>;
static_assert(std::same_as<decltype(std::views::repeat("foo", std::unreachable_sentinel)), RPV>);  // OK
static_assert(std::same_as<decltype(std::views::repeat(+"foo", std::unreachable_sentinel)), RPV>); // OK
static_assert(std::same_as<decltype(std::views::repeat("foo")), RPV>);                             // OK since LWG4054
static_assert(std::same_as<decltype(std::views::repeat(+"foo")), RPV>);                            // OK

constexpr bool test() {
  assert(*std::views::repeat(33).begin() == 33);
  assert(*std::views::repeat(33, 10).begin() == 33);
  static_assert(std::same_as<decltype(std::views::repeat(42)), std::ranges::repeat_view<int>>);
  static_assert(std::same_as<decltype(std::views::repeat(42, 3)), std::ranges::repeat_view<int, int>>);
  static_assert(std::same_as<decltype(std::views::repeat), decltype(std::ranges::views::repeat)>);

  // unbound && drop_view
  {
    auto r = std::views::repeat(33) | std::views::drop(3);
    static_assert(!std::ranges::sized_range<decltype(r)>);
    assert(*r.begin() == 33);
  }

  // bound && drop_view
  {
    auto r = std::views::repeat(33, 8) | std::views::drop(3);
    static_assert(std::ranges::sized_range<decltype(r)>);
    assert(*r.begin() == 33);
    assert(r.size() == 5);
  }

  // unbound && take_view
  {
    auto r = std::views::repeat(33) | std::views::take(3);
    static_assert(std::ranges::sized_range<decltype(r)>);
    assert(*r.begin() == 33);
    assert(r.size() == 3);
  }

  // bound && take_view
  {
    auto r = std::views::repeat(33, 8) | std::views::take(3);
    static_assert(std::ranges::sized_range<decltype(r)>);
    assert(*r.begin() == 33);
    assert(r.size() == 3);
  }

  // bound && transform_view
  {
    auto r = std::views::repeat(33, 8) | std::views::transform(Add3{});
    assert(*r.begin() == 36);
    assert(r.size() == 8);
  }

  // unbound && transform_view
  {
    auto r = std::views::repeat(33) | std::views::transform(Add3{});
    assert(*r.begin() == 36);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
