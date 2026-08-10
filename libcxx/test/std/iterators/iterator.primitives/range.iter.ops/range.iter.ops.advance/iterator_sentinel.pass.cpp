//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// ranges::advance(it, sent)

#include <iterator>

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <bool Count, class It>
constexpr void check_assignable(int* first, int* last, int* expected) {
  {
    It it(first);
    auto sent = assignable_sentinel(It(last));
    std::ranges::advance(it, sent);
    ASSERT_SAME_TYPE(decltype(std::ranges::advance(it, sent)), void);
    assert(base(it) == expected);
  }

  // Count operations
  if constexpr (Count) {
    IteratorOpCounts ops;
    auto it   = operation_counting_iterator(It(first), &ops);
    auto sent = assignable_sentinel(operation_counting_iterator(It(last), &ops));
    std::ranges::advance(it, sent);
    assert(base(base(it)) == expected);
    assert(ops.increments + ops.decrements == 0); // because we got here by assigning from last, not by incrementing
  }
}

template <bool Count, class It>
constexpr void check_sized_sentinel(int* first, int* last, int* expected) {
  auto size = (last - first);

  {
    It it(first);
    auto sent = distance_apriori_sentinel(size);
    std::ranges::advance(it, sent);
    ASSERT_SAME_TYPE(decltype(std::ranges::advance(it, sent)), void);
    assert(base(it) == expected);
  }

  // Count operations
  if constexpr (Count) {
    IteratorOpCounts ops;
    auto it   = operation_counting_iterator(It(first), &ops);
    auto sent = distance_apriori_sentinel(size);
    std::ranges::advance(it, sent);
    if constexpr (std::random_access_iterator<It>) {
      assert(ops.increments + ops.decrements + ops.zero_moves == 1);
    } else {
      const auto big   = std::max(ops.increments, ops.decrements);
      const auto small = std::min(ops.increments, ops.decrements);
      assert(big == static_cast<size_t>(size > 0 ? size : -size));
      assert(small == 0);
    }
  }
}

template <bool Count, class It>
constexpr void check_sentinel(int* first, int* last, int* expected) {
  auto size = (last - first);

  {
    It it(first);
    auto sent = sentinel_wrapper(It(last));
    std::ranges::advance(it, sent);
    ASSERT_SAME_TYPE(decltype(std::ranges::advance(it, sent)), void);
    assert(base(it) == expected);
  }

  // Count operations
  if constexpr (Count) {
    IteratorOpCounts ops;
    auto it   = operation_counting_iterator(It(first), &ops);
    auto sent = sentinel_wrapper(operation_counting_iterator(It(last), &ops));
    std::ranges::advance(it, sent);
    const auto big   = std::max(ops.increments, ops.decrements);
    const auto small = std::min(ops.increments, ops.decrements);
    assert(big == static_cast<size_t>(size > 0 ? size : -size));
    assert(small == 0);
  }
}

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (int n = 0; n != 10; ++n) {
    check_assignable<false, cpp17_input_iterator<int*>>(  range, range+n, range+n);
    check_assignable<false, cpp20_input_iterator<int*>>(  range, range+n, range+n);
    check_assignable<true,  forward_iterator<int*>>(      range, range+n, range+n);
    check_assignable<true,  bidirectional_iterator<int*>>(range, range+n, range+n);
    check_assignable<true,  random_access_iterator<int*>>(range, range+n, range+n);
    check_assignable<true,  contiguous_iterator<int*>>(   range, range+n, range+n);
    check_assignable<true,  int*>(                        range, range+n, range+n);

    check_sized_sentinel<false, cpp17_input_iterator<int*>>(  range, range+n, range+n);
    check_sized_sentinel<false, cpp20_input_iterator<int*>>(  range, range+n, range+n);
    check_sized_sentinel<true,  forward_iterator<int*>>(      range, range+n, range+n);
    check_sized_sentinel<true,  bidirectional_iterator<int*>>(range, range+n, range+n);
    check_sized_sentinel<true,  random_access_iterator<int*>>(range, range+n, range+n);
    check_sized_sentinel<true,  contiguous_iterator<int*>>(   range, range+n, range+n);
    check_sized_sentinel<true,  int*>(                        range, range+n, range+n);

    check_sentinel<false, cpp17_input_iterator<int*>>(  range, range+n, range+n);
    check_sentinel<false, cpp20_input_iterator<int*>>(  range, range+n, range+n);
    check_sentinel<true,  forward_iterator<int*>>(      range, range+n, range+n);
    check_sentinel<true,  bidirectional_iterator<int*>>(range, range+n, range+n);
    check_sentinel<true,  random_access_iterator<int*>>(range, range+n, range+n);
    check_sentinel<true,  contiguous_iterator<int*>>(   range, range+n, range+n);
    check_sentinel<true,  int*>(                        range, range+n, range+n);
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
