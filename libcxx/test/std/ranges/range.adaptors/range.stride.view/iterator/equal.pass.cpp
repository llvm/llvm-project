//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include <cassert>
#include <ranges>

#include "test_iterators.h"

template <class Iter>
constexpr void testOne() {
  using Range      = std::ranges::subrange<Iter>;
  using StrideView = std::ranges::stride_view<Range>;
  static_assert(std::ranges::common_range<StrideView>);

  {
    // simple test
    {
      int buffer[] = {0, 1, 2, -1, 4, 5, 6};
      Range input(Iter{buffer}, Iter{buffer + 7});
      StrideView sv(input, 1);
      StrideView sv_too(input, 2);
      auto b = sv.begin(), e = sv.end();
      auto b_too = sv_too.begin();

      assert(b == b);
      assert(!(b != b));

      assert(e == e);
      assert(!(e != e));

      assert(!(b == e));
      assert(b != e);

      std::advance(b, 8);
      std::advance(b_too, 4);

      assert(b == b_too);
      assert(!(b != b_too));

      assert(b == b);
      assert(!(b != b));

      assert(e == e);
      assert(!(e != e));

      assert(b == e);
      assert(!(b != e));
    }

    // Default-constructed iterators compare equal.
    {
      int buffer[] = {0, 1, 2, -1, 4, 5, 6};
      Range input(Iter{buffer}, Iter{buffer + 7});
      std::ranges::stride_view sv(input, 1);
      using StrideViewIter = decltype(sv.begin());
      StrideViewIter i1, i2;
      assert(i1 == i2);
      assert(!(i1 != i2));
    }
  }
}

constexpr bool test() {
  testOne<forward_iterator<int*>>();
  //testOne<bidirectional_iterator<int*>>();
  //testOne<random_access_iterator<int*>>();
  //testOne<contiguous_iterator<int*>>();
  testOne<int*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
