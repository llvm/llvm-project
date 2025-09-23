//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>
#include <cassert>
#include <initializer_list>

#include "test_iterators.h"
#include "test_range.h"

struct ForwardRange : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit ForwardRange(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

struct BidirectionalRange : std::ranges::view_base {
  using Iterator = bidirectional_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit BidirectionalRange(int* b, int* e): begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }
private:
  int* begin_;
  int* end_;
};

template <typename View>
constexpr void compareViews(View v, std::initializer_list<int> list) {
  auto b1 = v.begin();
  auto e1 = v.end();
  auto b2 = list.begin();
  auto e2 = list.end();
  for (; b1 != e1 && b2 != e2;) {
    assert(*b1 == *b2);
    ++b1;
    ++b2;
  }
  assert(b1 == e1);
  assert(b2 == e2);
}

constexpr bool test() {
  int arr[]  = {0, 1, 2, 3};
  int arr2[] = {4, 5, 6, 7};

  // one range
  {
    ForwardRange range(arr, arr + 4);

    {
      decltype(auto) result = std::views::concat(range);
      compareViews(result, {0, 1, 2, 3});
      ASSERT_SAME_TYPE(decltype(std::views::all(range)), decltype(result));
    }
  }

  // more than one range of same types
  {
    ForwardRange first(arr, arr + 4);
    ForwardRange tail(arr2, arr2 + 4);

    {
      decltype(auto) result = std::views::concat(first, tail);
      compareViews(result, {0, 1, 2, 3, 4, 5, 6, 7});
      using Type = std::ranges::concat_view<ForwardRange, ForwardRange>;
      ASSERT_SAME_TYPE(Type, decltype(result));
    }
  }

  // more than one range of different types
  {
    ForwardRange first(arr, arr + 4);
    BidirectionalRange tail(arr2, arr2 + 4);
    {
      decltype(auto) result = std::views::concat(first, tail);
      compareViews(result, {0, 1, 2, 3, 4, 5, 6, 7});
      using Type = std::ranges::concat_view<ForwardRange, BidirectionalRange>;
      ASSERT_SAME_TYPE(Type, decltype(result));
    }
  }

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
