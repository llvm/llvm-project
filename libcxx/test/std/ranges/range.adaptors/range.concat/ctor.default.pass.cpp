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

int buff[4] = {0, 1, 2, 3};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + 4) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

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
using BadView1    = std::ranges::concat_view<DefaultConstructibleView, NoDefaultView>;
using BadView2    = std::ranges::concat_view<NoDefaultView, NoDefaultView>;

constexpr bool test() {
  static_assert(std::is_default_constructible_v<DefaultView>);
  static_assert(!std::is_default_constructible_v<BadView1>);
  static_assert(!std::is_default_constructible_v<BadView2>);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
