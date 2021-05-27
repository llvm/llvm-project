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

// ranges::prev(iterator, count)

#include <iterator>

#include <array>
#include <cassert>

#include "check_round_trip.h"
#include "test_iterators.h"

using range_t = std::array<int, 10>;

template <std::input_or_output_iterator I>
constexpr void iterator_count_impl(I first, std::ptrdiff_t const n, range_t::const_iterator const expected) {
  auto result = std::ranges::prev(stride_counting_iterator(std::move(first)), n);
  assert(std::move(result).base().base() == expected);
  check_round_trip(result, n);
}

constexpr bool check_iterator_count() {
  constexpr auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  iterator_count_impl(bidirectional_iterator(&range[8]), 6, &range[2]);
  iterator_count_impl(random_access_iterator(&range[7]), 4, &range[3]);
  iterator_count_impl(contiguous_iterator(&range[5]), 5, &range[0]);

  iterator_count_impl(bidirectional_iterator(&range[2]), 0, &range[2]);
  iterator_count_impl(random_access_iterator(&range[3]), 0, &range[3]);
  iterator_count_impl(contiguous_iterator(&range[0]), 0, &range[0]);

  iterator_count_impl(bidirectional_iterator(&range[3]), -5, &range[8]);
  iterator_count_impl(random_access_iterator(&range[3]), -3, &range[6]);
  iterator_count_impl(contiguous_iterator(&range[3]), -1, &range[4]);
  return true;
}

int main(int, char**) {
  static_assert(check_iterator_count());
  check_iterator_count();
  return 0;
}
