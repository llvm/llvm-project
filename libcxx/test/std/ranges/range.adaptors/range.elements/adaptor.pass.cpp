//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::elements<N>
// std::views::keys
// std::views::values

#include <algorithm>
#include <cassert>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_range.h"

template <class T>
struct View : std::ranges::view_base {
  T* begin() const;
  T* end() const;
};

static_assert(!std::is_invocable_v<decltype((std::views::elements<0>))>);
static_assert(!std::is_invocable_v<decltype((std::views::elements<0>)), View<int>>);
static_assert(std::is_invocable_v<decltype((std::views::elements<0>)), View<std::pair<int, int>>>);
static_assert(std::is_invocable_v<decltype((std::views::elements<0>)), View<std::tuple<int>>>);
static_assert(!std::is_invocable_v<decltype((std::views::elements<5>)), View<std::tuple<int>>>);

static_assert(!std::is_invocable_v<decltype((std::views::keys))>);
static_assert(!std::is_invocable_v<decltype((std::views::keys)), View<int>>);
static_assert(std::is_invocable_v<decltype((std::views::keys)), View<std::pair<int, int>>>);
static_assert(std::is_invocable_v<decltype((std::views::keys)), View<std::tuple<int>>>);

static_assert(!std::is_invocable_v<decltype((std::views::values))>);
static_assert(!std::is_invocable_v<decltype((std::views::values)), View<int>>);
static_assert(std::is_invocable_v<decltype((std::views::values)), View<std::pair<int, int>>>);
static_assert(!std::is_invocable_v<decltype((std::views::values)), View<std::tuple<int>>>);

static_assert(!CanBePiped<View<int>, decltype((std::views::elements<0>))>);
static_assert(CanBePiped<View<std::pair<int, int>>, decltype((std::views::elements<0>))>);
static_assert(CanBePiped<View<std::tuple<int>>, decltype((std::views::elements<0>))>);
static_assert(!CanBePiped<View<std::tuple<int>>, decltype((std::views::elements<5>))>);

static_assert(!CanBePiped<View<int>, decltype((std::views::keys))>);
static_assert(CanBePiped<View<std::pair<int, int>>, decltype((std::views::keys))>);
static_assert(CanBePiped<View<std::tuple<int>>, decltype((std::views::keys))>);

static_assert(!CanBePiped<View<int>, decltype((std::views::values))>);
static_assert(CanBePiped<View<std::pair<int, int>>, decltype((std::views::values))>);
static_assert(!CanBePiped<View<std::tuple<int>>, decltype((std::views::values))>);

constexpr bool test() {
  std::pair<int, int> buff[] = {{1, 2}, {3, 4}, {5, 6}};

  // Test `views::elements<N>(v)`
  {
    using Result = std::ranges::elements_view<std::ranges::ref_view<std::pair<int, int>[3]>, 0>;
    std::same_as<Result> decltype(auto) result = std::views::elements<0>(buff);
    auto expected                              = {1, 3, 5};
    assert(std::ranges::equal(result, expected));
  }

  // Test `views::keys(v)`
  {
    using Result = std::ranges::elements_view<std::ranges::ref_view<std::pair<int, int>[3]>, 0>;
    std::same_as<Result> decltype(auto) result = std::views::keys(buff);
    auto expected                              = {1, 3, 5};
    assert(std::ranges::equal(result, expected));
  }

  // Test `views::values(v)`
  {
    using Result = std::ranges::elements_view<std::ranges::ref_view<std::pair<int, int>[3]>, 1>;
    std::same_as<Result> decltype(auto) result = std::views::values(buff);
    auto expected                              = {2, 4, 6};
    assert(std::ranges::equal(result, expected));
  }

  // Test `v | views::elements<N>`
  {
    using Result = std::ranges::elements_view<std::ranges::ref_view<std::pair<int, int>[3]>, 1>;
    std::same_as<Result> decltype(auto) result = buff | std::views::elements<1>;
    auto expected                              = {2, 4, 6};
    assert(std::ranges::equal(result, expected));
  }

  // Test `v | views::keys`
  {
    using Result = std::ranges::elements_view<std::ranges::ref_view<std::pair<int, int>[3]>, 0>;
    std::same_as<Result> decltype(auto) result = buff | std::views::keys;
    auto expected                              = {1, 3, 5};
    assert(std::ranges::equal(result, expected));
  }

  // Test `v | views::values`
  {
    using Result = std::ranges::elements_view<std::ranges::ref_view<std::pair<int, int>[3]>, 1>;
    std::same_as<Result> decltype(auto) result = buff | std::views::values;
    auto expected                              = {2, 4, 6};
    assert(std::ranges::equal(result, expected));
  }

  // Test views::elements<0> | views::elements<0>
  {
    std::pair<std::tuple<int>, std::tuple<int>> nested[] = {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}};
    using Result                                         = std::ranges::elements_view<
        std::ranges::elements_view<std::ranges::ref_view<std::pair<std::tuple<int>, std::tuple<int>>[3]>, 0>,
        0>;
    auto const partial                         = std::views::elements<0> | std::views::elements<0>;
    std::same_as<Result> decltype(auto) result = nested | partial;
    auto expected                              = {1, 3, 5};
    assert(std::ranges::equal(result, expected));
  }

  // Test views::keys | views::keys
  {
    std::pair<std::tuple<int>, std::tuple<int>> nested[] = {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}};
    using Result                                         = std::ranges::elements_view<
        std::ranges::elements_view<std::ranges::ref_view<std::pair<std::tuple<int>, std::tuple<int>>[3]>, 0>,
        0>;
    auto const partial                         = std::views::keys | std::views::keys;
    std::same_as<Result> decltype(auto) result = nested | partial;
    auto expected                              = {1, 3, 5};
    assert(std::ranges::equal(result, expected));
  }

  // Test views::values | views::values
  {
    std::pair<std::tuple<int>, std::tuple<int, int>> nested[] = {{{1}, {2, 3}}, {{4}, {5, 6}}, {{7}, {8, 9}}};
    using Result                                              = std::ranges::elements_view<
        std::ranges::elements_view<std::ranges::ref_view<std::pair<std::tuple<int>, std::tuple<int, int>>[3]>, 1>,
        1>;
    auto const partial                         = std::views::values | std::views::values;
    std::same_as<Result> decltype(auto) result = nested | partial;
    auto expected                              = {3, 6, 9};
    assert(std::ranges::equal(result, expected));
  }

  // Test views::keys | views::values
  {
    std::pair<std::tuple<int, int>, std::tuple<int>> nested[] = {{{1, 2}, {3}}, {{4, 5}, {6}}, {{7, 8}, {9}}};
    using Result                                              = std::ranges::elements_view<
        std::ranges::elements_view<std::ranges::ref_view<std::pair<std::tuple<int, int>, std::tuple<int>>[3]>, 0>,
        1>;
    auto const partial                         = std::views::keys | std::views::values;
    std::same_as<Result> decltype(auto) result = nested | partial;
    auto expected                              = {2, 5, 8};
    assert(std::ranges::equal(result, expected));
  }

  // Test views::values | views::keys
  {
    std::pair<std::tuple<int>, std::tuple<int, int>> nested[] = {{{1}, {2, 3}}, {{4}, {5, 6}}, {{7}, {8, 9}}};
    using Result                                              = std::ranges::elements_view<
        std::ranges::elements_view<std::ranges::ref_view<std::pair<std::tuple<int>, std::tuple<int, int>>[3]>, 1>,
        0>;
    auto const partial                         = std::views::values | std::views::keys;
    std::same_as<Result> decltype(auto) result = nested | partial;
    auto expected                              = {2, 5, 8};
    assert(std::ranges::equal(result, expected));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
