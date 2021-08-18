//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// constexpr decltype(auto) operator[](iter_difference_t<I> n) const
//   requires random_access_iterator<I>;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template<class Iter>
concept SubscriptEnabled = requires(Iter& iter) {
  iter[1];
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i)
      assert(iter[i - 1] == i);
  }
  {
    std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i)
      assert(iter[i - 1] == i);
  }

  {
    const std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
    assert(iter[0] == 1);
  }
  {
    const std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    assert(iter[7] == 8);
  }

  {
      static_assert( SubscriptEnabled<std::counted_iterator<random_access_iterator<int*>>>);
      static_assert(!SubscriptEnabled<std::counted_iterator<bidirectional_iterator<int*>>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
