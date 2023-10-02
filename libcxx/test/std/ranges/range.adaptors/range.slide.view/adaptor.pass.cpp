//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::views::slide

#include "test_iterators.h"
#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

struct SizedView : std::ranges::view_base {
  int* begin_ = nullptr;
  int* end_   = nullptr;
  constexpr SizedView(int* begin, int* end) : begin_(begin), end_(end) {}

  constexpr auto begin() const { return forward_iterator<int*>(begin_); }
  constexpr auto end() const { return sized_sentinel<forward_iterator<int*>>(forward_iterator<int*>(end_)); }
};
static_assert(std::ranges::forward_range<SizedView>);
static_assert(std::ranges::sized_range<SizedView>);
static_assert(std::ranges::view<SizedView>);

constexpr bool test() {
  constexpr int n = 8;
  int buf[n]      = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test that `std::views::slide` is a range adaptor.
  {
    using SomeView = SizedView;

    // Test `view | views::take`
    {
      SomeView view(buf, buf + n);
      std::same_as<std::ranges::slide_view<SomeView>> decltype(auto) result = view | std::views::slide(2);
      assert(result.base().begin_ == buf);
      assert(result.base().end_ == buf + n);
      assert(result.size() == 7);
    }
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
