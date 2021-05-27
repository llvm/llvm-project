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

// ranges::next(first, n)

#include <iterator>

#include <array>
#include <cassert>

#include "check_round_trip.h"
#include "test_iterators.h"

using range_t = std::array<int, 10>;

constexpr bool check_iterator() {
  constexpr auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  assert(std::ranges::next(cpp17_input_iterator(&range[0])) == cpp17_input_iterator(&range[1]));
  assert(std::ranges::next(cpp20_input_iterator(&range[1])).base() == &range[2]);
  assert(std::ranges::next(forward_iterator(&range[2])) == forward_iterator(&range[3]));
  assert(std::ranges::next(bidirectional_iterator(&range[3])) == bidirectional_iterator(&range[4]));
  assert(std::ranges::next(random_access_iterator(&range[4])) == random_access_iterator(&range[5]));
  assert(std::ranges::next(contiguous_iterator(&range[5])) == contiguous_iterator(&range[6]));
  assert(std::ranges::next(output_iterator(&range[6])).base() == &range[7]);
  return true;
}

int main(int, char**) {
  static_assert(check_iterator());
  check_iterator();
  return 0;
}
