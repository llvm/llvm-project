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

// ranges::next

#include <iterator>

#include <array>
#include <cassert>

#include "check_round_trip.h"
#include "test_iterators.h"

template <std::input_or_output_iterator I>
constexpr void check_iterator_count_sentinel_impl(I first, std::ptrdiff_t const steps, I const last) {
  auto result = std::ranges::next(stride_counting_iterator(first), steps, stride_counting_iterator(last));
  assert(result == last);
  check_round_trip(result, steps);
}

constexpr bool check_iterator_count_sentinel() {
  constexpr auto range = std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  check_iterator_count_sentinel_impl(cpp17_input_iterator(&range[0]), 1, cpp17_input_iterator(&range[1]));
  check_iterator_count_sentinel_impl(forward_iterator(&range[0]), 2, forward_iterator(&range[2]));
  check_iterator_count_sentinel_impl(bidirectional_iterator(&range[2]), 6, bidirectional_iterator(&range[8]));
  check_iterator_count_sentinel_impl(random_access_iterator(&range[3]), 2, random_access_iterator(&range[5]));
  check_iterator_count_sentinel_impl(contiguous_iterator(&range[0]), 5, contiguous_iterator(&range[5]));
  check_iterator_count_sentinel_impl(output_iterator(&range[3]), 2, output_iterator(&range[5]));

  check_iterator_count_sentinel_impl(cpp17_input_iterator(&range[0]), 0, cpp17_input_iterator(&range[0]));
  check_iterator_count_sentinel_impl(forward_iterator(&range[0]), 0, forward_iterator(&range[0]));
  check_iterator_count_sentinel_impl(bidirectional_iterator(&range[2]), 0, bidirectional_iterator(&range[2]));
  check_iterator_count_sentinel_impl(random_access_iterator(&range[3]), 0, random_access_iterator(&range[3]));
  check_iterator_count_sentinel_impl(contiguous_iterator(&range[0]), 0, contiguous_iterator(&range[0]));
  check_iterator_count_sentinel_impl(output_iterator(&range[3]), 0, output_iterator(&range[3]));

  check_iterator_count_sentinel_impl(bidirectional_iterator(&range[6]), -1, bidirectional_iterator(&range[5]));
  check_iterator_count_sentinel_impl(random_access_iterator(&range[7]), -2, random_access_iterator(&range[5]));
  check_iterator_count_sentinel_impl(contiguous_iterator(&range[8]), -3, contiguous_iterator(&range[5]));
  return true;
}

int main(int, char**) {
  static_assert(check_iterator_count_sentinel());
  assert(check_iterator_count_sentinel());

  return 0;
}
