//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr concat_view() = default;

#include <cassert>
#include <ranges>
#include <type_traits>
#include <vector>

constexpr int buff[4] = {0, 1, 2, 3};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + 4) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }
  constexpr auto size() const { return 4; }

private:
  int const* begin_;
  int const* end_;
};

struct NoDefaultView : std::ranges::view_base {
  NoDefaultView() = delete;
  int* begin() const;
  int* end() const;
};

using DefaultView = std::ranges::concat_view<DefaultConstructibleView, DefaultConstructibleView>;
using DefaultViewWithDiffTypes =
    std::ranges::concat_view<DefaultConstructibleView, decltype(std::views::all(std::declval<std::vector<int>>()))>;
using BadView1    = std::ranges::concat_view<DefaultConstructibleView, NoDefaultView>;
using BadView2    = std::ranges::concat_view<NoDefaultView, NoDefaultView>;

constexpr bool test() {
  static_assert(std::is_default_constructible_v<DefaultView>);
  static_assert(std::is_default_constructible_v<DefaultViewWithDiffTypes>);
  static_assert(!std::is_default_constructible_v<BadView1>);
  static_assert(!std::is_default_constructible_v<BadView2>);

  {
    DefaultView view = DefaultView();
    assert(view.size() == 8);
    auto it = view.begin();
    assert(*it++ == 0);
    assert(*it++ == 1);
    assert(*it++ == 2);
    assert(*it++ == 3);
  }

  {
    DefaultViewWithDiffTypes view = DefaultViewWithDiffTypes();
    assert(view.size() == 4);
    auto it = view.begin();
    assert(*it++ == 0);
    assert(*it++ == 1);
    assert(*it++ == 2);
    assert(*it++ == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
