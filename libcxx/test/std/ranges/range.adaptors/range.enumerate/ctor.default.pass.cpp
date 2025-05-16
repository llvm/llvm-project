//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// constexpr enumerate_view() requires default_initializable<V>;

#include <ranges>

#include <cassert>
#include <tuple>
#include <type_traits>

constexpr int buff[] = {0, 1};

template <bool DefaultConstructible>
struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView()
    requires DefaultConstructible
      : begin_(buff), end_(buff + 1) {}

  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

  int const* begin_;
  int const* end_;
};

static_assert(std::is_default_constructible_v<std::ranges::enumerate_view<DefaultConstructibleView<true>>>);
static_assert(!std::is_default_constructible_v<std::ranges::enumerate_view<DefaultConstructibleView<false>>>);

constexpr bool test() {
  using EnumerateView = std::ranges::enumerate_view<DefaultConstructibleView<true>>;

  {
    EnumerateView view;

    assert((*view.begin() == std::tuple<std::ranges::iterator_t<EnumerateView>::difference_type, int>{0, 0}));
    assert((*view.end() == std::tuple<std::ranges::iterator_t<EnumerateView>::difference_type, int>{1, 1}));

    auto [bi, bv] = *view.begin();
    assert(bi == 0);
    assert(bv == 0);

    auto [ei, ev] = *view.end();
    assert(ei == 1);
    assert(ev == 1);
  }
  {
    EnumerateView view = {};

    assert((*view.begin() == std::tuple<std::ranges::iterator_t<EnumerateView>::difference_type, int>{0, 0}));
    assert((*view.end() == std::tuple<std::ranges::iterator_t<EnumerateView>::difference_type, int>{1, 1}));

    auto [bi, bv] = *view.begin();
    assert(bi == 0);
    assert(bv == 0);

    auto [ei, ev] = *view.end();
    assert(ei == 1);
    assert(ev == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
