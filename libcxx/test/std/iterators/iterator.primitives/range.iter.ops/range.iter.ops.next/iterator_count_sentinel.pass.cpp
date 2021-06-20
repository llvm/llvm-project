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

// ranges::next(it, n, bound)

#include <iterator>
#include <cassert>

#include "test_iterators.h"

template <std::input_or_output_iterator It>
constexpr void check(It it, std::ptrdiff_t n, It last) {
  {
    It result = std::ranges::next(it, n, last);
    assert(result == last);
  }

  // Count the number of operations
  {
    stride_counting_iterator<It> strided_it(it);
    stride_counting_iterator<It> strided_last(last);
    stride_counting_iterator<It> result = std::ranges::next(strided_it, n, strided_last);
    assert(result == strided_last);
    if constexpr (std::random_access_iterator<It>) {
      if (n == 0 || n >= (last - it)) {
        assert(result.stride_count() == 0); // uses the assign-from-sentinel codepath
      } else {
        assert(result.stride_count() == 1); // uses += exactly once
      }
    } else {
      std::ptrdiff_t const abs_n = n < 0 ? -n : n;
      assert(result.stride_count() == abs_n);
      assert(result.stride_displacement() == n);
    }
  }
}

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  check(cpp17_input_iterator(&range[0]), 1, cpp17_input_iterator(&range[1]));
  check(forward_iterator(&range[0]), 2, forward_iterator(&range[2]));
  check(bidirectional_iterator(&range[2]), 6, bidirectional_iterator(&range[8]));
  check(random_access_iterator(&range[3]), 2, random_access_iterator(&range[5]));
  check(contiguous_iterator(&range[0]), 5, contiguous_iterator(&range[5]));

  check(cpp17_input_iterator(&range[0]), 0, cpp17_input_iterator(&range[0]));
  check(forward_iterator(&range[0]), 0, forward_iterator(&range[0]));
  check(bidirectional_iterator(&range[2]), 0, bidirectional_iterator(&range[2]));
  check(random_access_iterator(&range[3]), 0, random_access_iterator(&range[3]));
  check(contiguous_iterator(&range[0]), 0, contiguous_iterator(&range[0]));

  check(bidirectional_iterator(&range[6]), -1, bidirectional_iterator(&range[5]));
  check(random_access_iterator(&range[7]), -2, random_access_iterator(&range[5]));
  check(contiguous_iterator(&range[8]), -3, contiguous_iterator(&range[5]));
  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
