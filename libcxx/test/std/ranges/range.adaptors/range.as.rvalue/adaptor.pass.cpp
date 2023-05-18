//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::views::as_rvalue

#include <cassert>
#include <functional>
#include <ranges>
#include <vector>

#include "test_iterators.h"

struct DefaultConstructibleView : std::ranges::view_base {
  int i_;
  int* begin();
  int* end();
};

struct RValueView : std::ranges::view_base {};

template <class View, class T>
concept HasPipe = requires(View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

struct NoView {};
static_assert(std::is_invocable_v<decltype(std::views::as_rvalue), DefaultConstructibleView>);
static_assert(!std::is_invocable_v<decltype(std::views::as_rvalue)>);
static_assert(!std::is_invocable_v<decltype(std::views::as_rvalue), NoView>);
static_assert(HasPipe<DefaultConstructibleView&, decltype(std::views::as_rvalue)>);
static_assert(HasPipe<int (&)[10], decltype(std::views::as_rvalue)>);
static_assert(!HasPipe<int (&&)[10], decltype(std::views::as_rvalue)>);
static_assert(!HasPipe<NoView, decltype(std::views::as_rvalue)>);
static_assert(std::is_same_v<decltype(std::views::as_rvalue), decltype(std::ranges::views::as_rvalue)>);

struct move_iterator_range {
  constexpr std::move_iterator<int*> begin() const { return {}; }
  constexpr std::move_iterator<int*> end() const { return {}; }
};

static_assert(!std::ranges::view<move_iterator_range>);
static_assert(std::ranges::range<move_iterator_range>);

constexpr bool test() {
  { // view | views::as_rvalue
    DefaultConstructibleView v{{}, 3};
    std::same_as<std::ranges::as_rvalue_view<DefaultConstructibleView>> decltype(auto) view = v | std::views::as_rvalue;
    assert(view.base().i_ == 3);
  }

  { // adaptor | views::as_rvalue
    DefaultConstructibleView v{{}, 3};
    const auto partial = std::views::transform(std::identity{}) | std::views::as_rvalue;
    std::same_as<std::ranges::as_rvalue_view<
        std::ranges::transform_view<DefaultConstructibleView, std::identity>>> decltype(auto) view = partial(v);
    assert(view.base().base().i_ == 3);
  }

  { // views::as_rvalue | adaptor
    DefaultConstructibleView v{{}, 3};
    const auto partial = std::views::as_rvalue | std::views::transform(std::identity{});
    std::same_as<std::ranges::transform_view<std::ranges::as_rvalue_view<DefaultConstructibleView>,
                                             std::identity>> decltype(auto) view = partial(v);
    assert(view.base().base().i_ == 3);
  }

  { // rvalue-view | views::as_rvalue
    int a[4] = {1, 2, 3, 4};
    std::ranges::subrange range(rvalue_iterator{a}, rvalue_iterator{a + 4});
    [[maybe_unused]] std::same_as<std::ranges::subrange<rvalue_iterator<int>>> decltype(auto) rval_range =
        range | std::views::as_rvalue;
  }

  { // range | views::as_rvalue
    [[maybe_unused]] std::same_as<std::ranges::as_rvalue_view<std::views::all_t<std::vector<int>>>> decltype(auto)
        view = std::vector<int>{} | std::views::as_rvalue;
  }

  { // rvalue-range | views::as_rvalue
    [[maybe_unused]] std::same_as<std::views::all_t<move_iterator_range>> decltype(auto) view =
        move_iterator_range{} | std::views::as_rvalue;
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
