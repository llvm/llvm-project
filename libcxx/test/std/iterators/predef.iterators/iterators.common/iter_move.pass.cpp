//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// friend iter_rvalue_reference_t<I> iter_move(const common_iterator& i)
//   noexcept(noexcept(ranges::iter_move(declval<const I&>())))
//     requires input_iterator<I>;

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "types.h"

void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1 = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    assert(std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(commonIter1)), int&&);
  }
  {
    auto iter1 = forward_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    assert(std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(commonIter1)), int&&);
  }
  {
    auto iter1 = random_access_iterator<int*>(buffer);
    auto commonIter1 = std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    assert(std::ranges::iter_move(commonIter1) == 1);
    ASSERT_SAME_TYPE(decltype(std::ranges::iter_move(commonIter1)), int&&);
  }
}

int main(int, char**) {
  test();

  return 0;
}
