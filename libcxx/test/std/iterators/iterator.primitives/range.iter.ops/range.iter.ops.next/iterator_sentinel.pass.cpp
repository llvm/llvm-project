//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::next(it, bound)

#include <iterator>

#include <array>
#include <cassert>
#include <utility>

#include "test_iterators.h"
#include "../types.h"

using range_t = std::array<int, 10>;

template <bool Count, typename It>
constexpr void check_assignable(int* it, int* last, int const* expected) {
  {
    It result = std::ranges::next(It(it), assignable_sentinel(It(last)));
    assert(base(result) == expected);
  }

  // Count operations
  if constexpr (Count) {
    auto strided_it = stride_counting_iterator(It(it));
    auto strided_last = assignable_sentinel(stride_counting_iterator(It(last)));
    stride_counting_iterator<It> result = std::ranges::next(std::move(strided_it), std::move(strided_last));
    assert(base(result.base()) == expected);
    assert(result.stride_count() == 0); // because we got here by assigning from last, not by incrementing
  }
}

template <typename It>
constexpr void check_sized_sentinel(int* it, int* last, int const* expected) {
  auto n = (last - it);

  {
    auto sent = distance_apriori_sentinel(n);
    auto result = std::ranges::next(It(it), sent);
    assert(base(result) == expected);
  }

  // Count operations
  {
    auto strided_it = stride_counting_iterator(It(it));
    auto sent = distance_apriori_sentinel(n);
    auto result = std::ranges::next(std::move(strided_it), sent);
    assert(base(result.base()) == expected);

    if constexpr (std::random_access_iterator<It>) {
      assert(result.stride_count() == 1); // should have used exactly one +=
      assert(result.stride_displacement() == 1);
    } else {
      assert(result.stride_count() == n);
      assert(result.stride_displacement() == n);
    }
  }
}

template <bool Count, typename It>
constexpr void check_sentinel(int* it, int* last, int const* expected) {
  auto n = (last - it);

  {
    auto sent = sentinel_wrapper(It(last));
    It result = std::ranges::next(It(it), sent);
    assert(base(result) == expected);
  }

  // Count operations
  if constexpr (Count) {
    auto strided_it = stride_counting_iterator(It(it));
    auto sent = sentinel_wrapper(stride_counting_iterator(It(last)));
    stride_counting_iterator result = std::ranges::next(std::move(strided_it), sent);
    assert(base(result.base()) == expected);
    assert(result.stride_count() == n); // must have used ++ until it hit the sentinel
  }
}

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  check_assignable<false, cpp17_input_iterator<int*>>(  &range[0], &range[2], &range[2]);
  check_assignable<true,  forward_iterator<int*>>(      &range[0], &range[3], &range[3]);
  check_assignable<true,  bidirectional_iterator<int*>>(&range[0], &range[4], &range[4]);
  check_assignable<true,  random_access_iterator<int*>>(&range[0], &range[5], &range[5]);
  check_assignable<true,  contiguous_iterator<int*>>(   &range[0], &range[6], &range[6]);

  check_sized_sentinel<cpp17_input_iterator<int*>>(  &range[0], &range[7], &range[7]);
  check_sized_sentinel<cpp20_input_iterator<int*>>(  &range[0], &range[6], &range[6]);
  check_sized_sentinel<forward_iterator<int*>>(      &range[0], &range[5], &range[5]);
  check_sized_sentinel<bidirectional_iterator<int*>>(&range[0], &range[4], &range[4]);
  check_sized_sentinel<random_access_iterator<int*>>(&range[0], &range[3], &range[3]);
  check_sized_sentinel<contiguous_iterator<int*>>(   &range[0], &range[2], &range[2]);

  check_sentinel<false, cpp17_input_iterator<int*>>(  &range[0], &range[1], &range[1]);
  // cpp20_input_iterator not copyable, so is omitted
  check_sentinel<true,  forward_iterator<int*>>(      &range[0], &range[3], &range[3]);
  check_sentinel<true,  bidirectional_iterator<int*>>(&range[0], &range[4], &range[4]);
  check_sentinel<true,  random_access_iterator<int*>>(&range[0], &range[5], &range[5]);
  check_sentinel<true,  contiguous_iterator<int*>>(   &range[0], &range[6], &range[6]);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
