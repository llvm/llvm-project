//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto operator->() const noexcept
//   requires contiguous_iterator<I>;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template<class Iter>
concept ArrowEnabled = requires(Iter& iter) {
  iter.operator->();
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    for (int i = 0; i < 8; ++i, ++iter)
      assert(iter.operator->() == buffer + i);

    static_assert(noexcept(iter.operator->()));
  }
  {
    const std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    assert(iter.operator->() == buffer);

    static_assert(noexcept(iter.operator->()));
  }

  {
      static_assert( ArrowEnabled<std::counted_iterator<contiguous_iterator<int*>>>);
      static_assert(!ArrowEnabled<std::counted_iterator<random_access_iterator<int*>>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
