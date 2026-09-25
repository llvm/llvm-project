//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// std::views::as_const

#include <array>
#include <cassert>
#include <functional>
#include <ranges>
#include <vector>

#include "test_iterators.h"

template <class View, class T>
concept HasPipe = requires {
  { std::declval<View>() | std::declval<T>() };
};

struct DefaultConstructibleView : std::ranges::view_base {
  int i_;
  int* begin();
  int* end();
};
struct NoView {};

static_assert(std::is_invocable_v<decltype(std::views::as_const), DefaultConstructibleView>);
static_assert(!std::is_invocable_v<decltype(std::views::as_const)>);
static_assert(!std::is_invocable_v<decltype(std::views::as_const), NoView>);
static_assert(HasPipe<DefaultConstructibleView&, decltype(std::views::as_const)>);
static_assert(HasPipe<int (&)[10], decltype(std::views::as_const)>);
static_assert(!HasPipe<int (&&)[10], decltype(std::views::as_const)>);
static_assert(!HasPipe<NoView, decltype(std::views::as_const)>);
static_assert(std::is_same_v<decltype(std::views::as_const), decltype(std::ranges::views::as_const)>);

struct const_iterator_range {
  constexpr std::const_iterator<int*> begin() const { return {}; }
  constexpr std::const_iterator<int*> end() const { return {}; }
};
static_assert(!std::ranges::view<const_iterator_range>);
static_assert(std::ranges::range<const_iterator_range>);

constexpr bool test() {
  // Let E be an expression, let T be decltype((E)), and let U be remove_cvref_t<T>.
  // The expression views::as_const(E) is expression-equivalent to:

  // - If views::all_t<T> models constant_range, then views::all(E).
  {
    [[maybe_unused]] std::same_as<std::views::all_t<const_iterator_range>> decltype(auto) view =
        std::views::as_const(const_iterator_range{});
  }
  {
    // ambiguous with empty_view case
    [[maybe_unused]] std::same_as<std::views::all_t<std::ranges::empty_view<const int>>> decltype(auto) view =
        std::views::empty<const int> | std::views::as_const;
  }
  {
    // ambiguous with span case
    int a[3] = {};
    [[maybe_unused]] std::same_as<std::views::all_t<std::span<const int>>> decltype(auto) view1 =
        std::span<const int>(a) | std::views::as_const;
    [[maybe_unused]] std::same_as<std::views::all_t<std::span<const int, 3>>> decltype(auto) view2 =
        std::span<const int, 3>(a) | std::views::as_const;
  }
  {
    // ambiguous with ref_view case
    std::array<int, 3> a                              = {};
    std::ranges::ref_view<const std::array<int, 3>> r = a;
    [[maybe_unused]] std::same_as<std::ranges::ref_view<const std::array<int, 3>>> decltype(auto) view =
        r | std::views::as_const;
  }
  {
    // ambiguous with constant_range case
    std::array<const int, 3> a = {};
    [[maybe_unused]] std::same_as<std::ranges::ref_view<std::array<const int, 3>>> decltype(auto) view =
        a | std::views::as_const;
  }

  // - Otherwise, if U denotes empty_view<X> for some type X, then auto(views::empty<const X>).
  {
    [[maybe_unused]] std::same_as<std::ranges::empty_view<const int>> decltype(auto) view =
        std::views::empty<int> | std::views::as_const;
  }

  // - Otherwise, if U denotes span<X, Extent> for some type X and some extent Extent, then span<const X, Extent>(E).
  {
    int a[3]                                               = {};
    std::same_as<std::span<const int>> decltype(auto) view = std::span<int>(a) | std::views::as_const;
    assert(std::to_address(view.begin()) == a);
    assert(std::to_address(view.end()) == a + 3);
  }
  {
    int a[3]                                                  = {};
    std::same_as<std::span<const int, 3>> decltype(auto) view = std::span<int, 3>(a) | std::views::as_const;
    assert(std::to_address(view.begin()) == a);
    assert(std::to_address(view.end()) == a + 3);
  }

  // - Otherwise, if U denotes ref_view<X> for some type X and const X models constant_range, then ref_view(static_cast<const X&>(E.base())).
  {
    std::array<int, 3> a                        = {};
    std::ranges::ref_view<std::array<int, 3>> r = a;
    [[maybe_unused]] std::same_as<std::ranges::ref_view<const std::array<int, 3>>> decltype(auto) view =
        r | std::views::as_const;
  }

  // - Otherwise, if E is an lvalue, const U models constant_range, and U does not model view, then ref_view(static_cast<const U&>(E)).
  {
    std::array<int, 3> a = {};
    [[maybe_unused]] std::same_as<std::ranges::ref_view<const std::array<int, 3>>> decltype(auto) view =
        a | std::views::as_const;
  }

  // - Otherwise, as_const_view(E).
  { // view | views::as_const
    DefaultConstructibleView v{{}, 3};
    std::same_as<std::ranges::as_const_view<DefaultConstructibleView>> decltype(auto) view = v | std::views::as_const;
    assert(view.base().i_ == 3);
  }

  { // adaptor | views::as_const
    DefaultConstructibleView v{{}, 3};
    const auto partial = std::views::transform(std::identity{}) | std::views::as_const;
    std::same_as<std::ranges::as_const_view<
        std::ranges::transform_view<DefaultConstructibleView, std::identity>>> decltype(auto) view = partial(v);
    assert(view.base().base().i_ == 3);
  }

  { // views::as_const | adaptor
    DefaultConstructibleView v{{}, 3};
    const auto partial = std::views::as_const | std::views::transform(std::identity{});
    std::same_as<std::ranges::transform_view<std::ranges::as_const_view<DefaultConstructibleView>,
                                             std::identity>> decltype(auto) view = partial(v);
    assert(view.base().base().i_ == 3);
  }

  { // range | views::as_const
    [[maybe_unused]] std::same_as<std::ranges::as_const_view<std::views::all_t<std::vector<int>>>> decltype(auto) view =
        std::vector<int>{} | std::views::as_const;
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
