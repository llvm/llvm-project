//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr bool operator==(const iterator& x, const sentinel& y);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <ranges>

#include "test_iterators.h"

template <class Iter>
constexpr void testOne() {
  using Sent      = sentinel_wrapper<Iter>;
  using Range     = std::ranges::subrange<Iter, Sent>;
  using SplitView = std::ranges::split_view<Range, std::ranges::single_view<int>>;
  static_assert(!std::ranges::common_range<SplitView>);

  {
    // simple test
    {
      int buffer[] = {0, 1, 2, -1, 4, 5, 6};
      Range input(Iter{buffer}, Sent{Iter{buffer + 7}});
      SplitView sv(input, -1);
      auto b = sv.begin();
      auto e = sv.end();

      assert(!(b == e));
      assert(b != e);

      std::advance(b, 2);
      assert(b == e);
      assert(!(b != e));
    }

    // iterator at trailing empty position should not equal to end
    {
      int buffer[] = {0, 1, 2, -1};
      Range input(Iter{buffer}, Sent{Iter{buffer + 4}});
      SplitView sv(input, -1);
      auto b = sv.begin();
      auto e = sv.end();

      ++b; // cur points to end but trailing_empty is true

      assert(b != e);
      assert(!(b == e));

      ++b;
      assert(b == e);
      assert(!(b != e));
    }
  }
}

constexpr bool test() {
  testOne<forward_iterator<int*>>();
  testOne<bidirectional_iterator<int*>>();
  testOne<random_access_iterator<int*>>();
  testOne<contiguous_iterator<int*>>();
  testOne<int*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
