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

// ranges::prev(it, n, bound)

#include <iterator>
#include <cassert>

#include "test_iterators.h"

template <std::bidirectional_iterator It>
constexpr void check(It it, std::ptrdiff_t n, It last) {
  auto abs = [](auto x) { return x < 0 ? -x : x; };

  {
    It result = std::ranges::prev(it, n, last);
    assert(result == last);
  }

  // Count the number of operations
  {
    stride_counting_iterator<It> strided_it(it);
    stride_counting_iterator<It> strided_last(last);
    stride_counting_iterator<It> result = std::ranges::prev(strided_it, n, strided_last);
    assert(result == strided_last);
    if constexpr (std::random_access_iterator<It>) {
      if (n == 0 || abs(n) >= abs(last - it)) {
        assert(result.stride_count() == 0); // uses the assign-from-sentinel codepath
      } else {
        assert(result.stride_count() == 1); // uses += exactly once
      }
    } else {
      assert(result.stride_count() == abs(n));
      assert(result.stride_displacement() == -n);
    }
  }
}

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  check(bidirectional_iterator(&range[8]), 6, bidirectional_iterator(&range[2]));
  check(random_access_iterator(&range[5]), 2, random_access_iterator(&range[3]));
  check(contiguous_iterator(&range[5]), 5, contiguous_iterator(&range[0]));

  check(bidirectional_iterator(&range[2]), 0, bidirectional_iterator(&range[2]));
  check(random_access_iterator(&range[3]), 0, random_access_iterator(&range[3]));
  check(contiguous_iterator(&range[0]), 0, contiguous_iterator(&range[0]));

  check(bidirectional_iterator(&range[5]), -1, bidirectional_iterator(&range[6]));
  check(random_access_iterator(&range[5]), -2, random_access_iterator(&range[7]));
  check(contiguous_iterator(&range[5]), -3, contiguous_iterator(&range[8]));
  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
