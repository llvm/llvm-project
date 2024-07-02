//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr bool operator==(__iterator const& __x, default_sentinel_t)
// friend constexpr bool operator==(__iterator const& __x, __iterator const& __y)

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "../types.h"

template <class Iter>
constexpr void testOne() {
  using Range = BasicTestView<Iter, Iter>;
  static_assert(std::ranges::common_range<Range>);
  using StrideView = std::ranges::stride_view<Range>;

  {
    // simple test
    {
      int buffer[] = {0, 1, 2, -1, 4, 5, 6, 7};
      const Range input(Iter{buffer}, Iter{buffer + 8});
      const StrideView sv(input, 1);
      const StrideView sv_too(input, 2);
      auto b     = sv.begin();
      auto e     = sv.end();
      auto b_too = sv_too.begin();

      assert(b == b);
      assert(!(b != b));

      // When Range is a bidirectional_range, the type of e is
      // default_sentinel_t and those do not compare to one another.
      if constexpr (!std::ranges::bidirectional_range<Range>) {
        assert(e == e);
        assert(!(e != e));
      }
      assert(!(b == e));
      assert(b != e);

      std::advance(b, 8);
      std::advance(b_too, 4);

      assert(b == b_too);
      assert(!(b != b_too));

      assert(b == b);
      assert(!(b != b));

      // See above.
      if constexpr (!std::ranges::bidirectional_range<Range>) {
        assert(e == e);
        assert(!(e != e));
      }

      assert(b == e);
      assert(!(b != e));
    }

    // Default-constructed iterators compare equal.
    {
      int buffer[] = {0, 1, 2, -1, 4, 5, 6};
      const Range input(Iter{buffer}, Iter{buffer + 7});
      const std::ranges::stride_view sv(input, 1);
      using StrideViewIter = decltype(sv.begin());
      StrideViewIter i1;
      StrideViewIter i2;
      assert(i1 == i2);
      assert(!(i1 != i2));
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
