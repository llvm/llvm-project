//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class... _Views>
// concat_view(_Views&&...) -> concat_view<views::all_t<_Views>...>;

#include <cassert>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  View() = default;
  forward_iterator<int*> begin() const;
  sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(std::ranges::view<View>);

// A range that is not a view
struct Range {
  Range() = default;
  forward_iterator<int*> begin() const;
  sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(std::ranges::range<Range> && !std::ranges::view<Range>);

constexpr bool test() {
  {
    View v;
    std::ranges::concat_view view(v);
    static_assert(std::is_same_v<decltype(view), std::ranges::concat_view<View>>);
  }

  // Test with a range that isn't a view, to make sure we properly use views::all_t in the implementation.
  {
    Range r;
    std::ranges::concat_view view(r);
    static_assert(std::is_same_v<decltype(view), std::ranges::concat_view<std::ranges::ref_view<Range>>>);
  }

  // Test a view which has a range and a view
  {
    Range r;
    View v;
    std::ranges::concat_view view(r, v);
    static_assert(std::is_same_v<decltype(view), std::ranges::concat_view<std::ranges::ref_view<Range>, View>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
