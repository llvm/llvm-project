//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// template <class _View>
// chunk_view(_View&&, range_difference_t<_View>) -> chunk_view<views::all_t<_View>>;

#include <ranges>

#include <cassert>
#include <type_traits>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  View() = default;
  forward_iterator<int*> begin() const;
  sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(std::ranges::view<View>);

struct Range {
  Range() = default;
  forward_iterator<int*> begin() const;
  sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(std::ranges::range<Range>);
static_assert(!std::ranges::view<Range>);

constexpr bool test() {
  {
    View v;
    std::ranges::chunk_view view(v, 42);
    static_assert(std::is_same_v<decltype(view), std::ranges::chunk_view<View>>);
  }
  {
    Range r;
    std::ranges::chunk_view view(r, 42);
    static_assert(std::is_same_v<decltype(view), std::ranges::chunk_view<std::ranges::ref_view<Range>>>);
  }
  {
    std::ranges::chunk_view view(Range{}, 42);
    static_assert(std::is_same_v<decltype(view), std::ranges::chunk_view<std::ranges::owning_view<Range>>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
