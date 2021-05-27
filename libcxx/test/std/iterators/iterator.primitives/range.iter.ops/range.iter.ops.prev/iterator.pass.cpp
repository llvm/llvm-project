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

// ranges::prev(iterator)

#include <iterator>

#include <array>
#include <cassert>

#include "check_round_trip.h"
#include "test_iterators.h"

constexpr bool check_iterator() {
  constexpr auto range = std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  assert(std::ranges::prev(bidirectional_iterator(&range[4])) == bidirectional_iterator(&range[3]));
  assert(std::ranges::prev(random_access_iterator(&range[5])) == random_access_iterator(&range[4]));
  assert(std::ranges::prev(contiguous_iterator(&range[6])) == contiguous_iterator(&range[5]));
  return true;
}

int main(int, char**) {
  static_assert(check_iterator());
  check_iterator();
  return 0;
}
