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

// ranges::next(it, n)

#include <iterator>

#include <cassert>
#include <utility>

#include "test_iterators.h"

template <std::input_or_output_iterator It>
constexpr void check_steps(It it, std::ptrdiff_t n, int const* expected) {
  {
    It result = std::ranges::next(std::move(it), n);
    assert(&*result == expected);
  }

  // Count the number of operations
  {
    stride_counting_iterator strided_it(std::move(it));
    stride_counting_iterator result = std::ranges::next(std::move(strided_it), n);
    assert(&*result == expected);
    if constexpr (std::random_access_iterator<It>) {
      assert(result.stride_count() == 1); // uses += exactly once
      assert(result.stride_displacement() == 1);
    } else {
      auto const abs_n = n < 0 ? -n : n;
      assert(result.stride_count() == abs_n);
      assert(result.stride_displacement() == n);
    }
  }
}

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  check_steps(cpp17_input_iterator(&range[0]), 1, &range[1]);
  check_steps(cpp20_input_iterator(&range[6]), 2, &range[8]);
  check_steps(forward_iterator(&range[0]), 3, &range[3]);
  check_steps(bidirectional_iterator(&range[2]), 6, &range[8]);
  check_steps(random_access_iterator(&range[3]), 4, &range[7]);
  check_steps(contiguous_iterator(&range[0]), 5, &range[5]);
  check_steps(output_iterator(&range[0]), 6, &range[6]);

  check_steps(cpp17_input_iterator(&range[0]), 0, &range[0]);
  check_steps(cpp20_input_iterator(&range[6]), 0, &range[6]);
  check_steps(forward_iterator(&range[0]), 0, &range[0]);
  check_steps(bidirectional_iterator(&range[2]), 0, &range[2]);
  check_steps(random_access_iterator(&range[3]), 0, &range[3]);
  check_steps(contiguous_iterator(&range[0]), 0, &range[0]);
  check_steps(output_iterator(&range[0]), 0, &range[0]);

  check_steps(bidirectional_iterator(&range[8]), -5, &range[3]);
  check_steps(random_access_iterator(&range[6]), -3, &range[3]);
  check_steps(contiguous_iterator(&range[4]), -1, &range[3]);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
