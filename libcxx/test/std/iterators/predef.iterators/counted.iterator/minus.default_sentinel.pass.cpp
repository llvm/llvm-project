//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// friend constexpr iter_difference_t<I> operator-(
//   const counted_iterator& x, default_sentinel_t);
// friend constexpr iter_difference_t<I> operator-(
//   default_sentinel_t, const counted_iterator& y);

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
    assert(iter - std::default_sentinel == -8);
    assert(std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - std::default_sentinel), std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(std::default_sentinel - iter), std::iter_difference_t<int*>);
  }
  {
    const std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
    assert(iter - std::default_sentinel == -8);
    assert(std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - std::default_sentinel), std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(std::default_sentinel - iter), std::iter_difference_t<int*>);
  }
  {
    std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    assert(iter - std::default_sentinel == -8);
    assert(std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - std::default_sentinel), std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(std::default_sentinel - iter), std::iter_difference_t<int*>);
  }
  {
    const std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    assert(iter - std::default_sentinel == -8);
    assert(std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - std::default_sentinel), std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(std::default_sentinel - iter), std::iter_difference_t<int*>);
  }
  {
    std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(iter - std::default_sentinel == -8);
    assert(std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - std::default_sentinel), std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(std::default_sentinel - iter), std::iter_difference_t<int*>);
  }
  {
    const std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(iter - std::default_sentinel == -8);
    assert(std::default_sentinel - iter == 8);
    assert(iter.count() == 8);

    ASSERT_SAME_TYPE(decltype(iter - std::default_sentinel), std::iter_difference_t<int*>);
    ASSERT_SAME_TYPE(decltype(std::default_sentinel - iter), std::iter_difference_t<int*>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
