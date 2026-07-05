//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// template <class Range, class Pred>
// chunk_by_view(Range&&, Pred) -> chunk_by_view<views::all_t<Range>, Pred>;

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

// A range that is not a view
struct Range {
  Range() = default;
  forward_iterator<int*> begin() const;
  sentinel_wrapper<forward_iterator<int*>> end() const;
};
static_assert(std::ranges::range<Range>);
static_assert(!std::ranges::view<Range>);

struct Pred {
  constexpr bool operator()(int x, int y) const { return x <= y; }
};

constexpr bool test() {
  {
    View v;
    Pred pred;
    std::ranges::chunk_by_view view(v, pred);
    static_assert(std::is_same_v<decltype(view), std::ranges::chunk_by_view<View, Pred>>);
  }
  {
    Range r;
    Pred pred;
    std::ranges::chunk_by_view view(r, pred);
    static_assert(std::is_same_v<decltype(view), std::ranges::chunk_by_view<std::ranges::ref_view<Range>, Pred>>);
  }
  {
    Pred pred;
    std::ranges::chunk_by_view view(Range{}, pred);
    static_assert(std::is_same_v<decltype(view), std::ranges::chunk_by_view<std::ranges::owning_view<Range>, Pred>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
