//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::views::zip_transform

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <type_traits>
#include <vector>

#include "types.h"

struct NotMoveConstructible {
  NotMoveConstructible()                       = default;
  NotMoveConstructible(NotMoveConstructible&&) = delete;
  int operator()() const { return 5; }
};

struct NotCopyConstructible {
  NotCopyConstructible()                            = default;
  NotCopyConstructible(NotCopyConstructible&&)      = default;
  NotCopyConstructible(const NotCopyConstructible&) = delete;
  int operator()() const { return 5; }
};

struct NotInvocable {};

template <class... Args>
struct Invocable {
  int operator()(Args...) const { return 5; }
};

struct ReturnNotObject {
  void operator()() const {}
};

// LWG3773 views::zip_transform still requires F to be copy_constructible when empty pack
static_assert(std::is_invocable_v<decltype((std::views::zip_transform)), NotCopyConstructible>);

static_assert(!std::is_invocable_v<decltype((std::views::zip_transform))>);
static_assert(!std::is_invocable_v<decltype((std::views::zip_transform)), NotMoveConstructible>);
static_assert(!std::is_invocable_v<decltype((std::views::zip_transform)), NotInvocable>);
static_assert(std::is_invocable_v<decltype((std::views::zip_transform)), Invocable<>>);
static_assert(!std::is_invocable_v<decltype((std::views::zip_transform)), ReturnNotObject>);

static_assert(std::is_invocable_v<decltype((std::views::zip_transform)), //
                                  Invocable<int>,                        //
                                  std::ranges::iota_view<int, int>>);
static_assert(!std::is_invocable_v<decltype((std::views::zip_transform)), //
                                   Invocable<>,                           //
                                   std::ranges::iota_view<int, int>>);
static_assert(!std::is_invocable_v<decltype((std::views::zip_transform)),
                                   Invocable<int>,
                                   std::ranges::iota_view<int, int>,
                                   std::ranges::iota_view<int, int>>);
static_assert(std::is_invocable_v<decltype((std::views::zip_transform)),
                                  Invocable<int, int>,
                                  std::ranges::iota_view<int, int>,
                                  std::ranges::iota_view<int, int>>);

constexpr bool test() {
  {
    // zip_transform function with no ranges
    auto v = std::views::zip_transform(Invocable<>{});
    assert(std::ranges::empty(v));
    static_assert(std::is_same_v<decltype(v), std::ranges::empty_view<int>>);
  }

  {
    // zip_transform views
    int buffer1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int buffer2[] = {9, 10, 11, 12};
    auto view1    = std::views::all(buffer1);
    auto view2    = std::views::all(buffer2);
    std::same_as<std::ranges::zip_transform_view<std::plus<>, decltype(view1), decltype(view2)>> decltype(auto) v =
        std::views::zip_transform(std::plus{}, buffer1, buffer2);
    assert(std::ranges::size(v) == 4);
    auto expected = {10, 12, 14, 16};
    assert(std::ranges::equal(v, expected));
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v)>, int>);
  }

  {
    // zip_transform a viewable range
    std::array a{1, 2, 3};
    auto id = [](auto& x) -> decltype(auto) { return (x); };
    std::same_as<
        std::ranges::zip_transform_view<decltype(id), std::ranges::ref_view<std::array<int, 3>>>> decltype(auto) v =
        std::views::zip_transform(id, a);
    assert(&v[0] == &a[0]);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v)>, int&>);
  }

  int buffer[] = {1, 2, 3};
  {
    // one range
    auto v = std::views::zip_transform(MakeTuple{}, SimpleCommon{buffer});
    assert(std::ranges::equal(v, std::vector{std::tuple(1), std::tuple(2), std::tuple(3)}));
  }

  {
    // two ranges
    auto v = std::views::zip_transform(GetFirst{}, SimpleCommon{buffer}, std::views::iota(0));
    assert(std::ranges::equal(v, std::vector{1, 2, 3}));
  }

  {
    // three ranges
    auto v = std::views::zip_transform(Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::single_view(2.));
    assert(std::ranges::equal(v, std::vector{std::tuple(1, 1, 2.0)}));
  }

  {
    // single empty range
    auto v = std::views::zip_transform(MakeTuple{}, std::ranges::empty_view<int>());
    assert(std::ranges::empty(v));
  }

  {
    // empty range at the beginning
    auto v = std::views::zip_transform(
        MakeTuple{}, std::ranges::empty_view<int>(), SimpleCommon{buffer}, SimpleCommon{buffer});
    assert(std::ranges::empty(v));
  }

  {
    // empty range in the middle
    auto v = std::views::zip_transform(
        MakeTuple{}, SimpleCommon{buffer}, std::ranges::empty_view<int>(), SimpleCommon{buffer});
    assert(std::ranges::empty(v));
  }

  {
    // empty range at the end
    auto v = std::views::zip_transform(
        MakeTuple{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::empty_view<int>());
    assert(std::ranges::empty(v));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
