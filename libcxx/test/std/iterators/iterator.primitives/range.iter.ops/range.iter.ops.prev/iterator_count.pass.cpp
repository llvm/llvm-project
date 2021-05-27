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

// ranges::prev(it, n)

#include <iterator>
#include <cassert>
#include <utility>

#include "test_iterators.h"

template <std::input_or_output_iterator I>
constexpr void check(I it, std::ptrdiff_t n, int const* expected) {
  auto result = std::ranges::prev(stride_counting_iterator(std::move(it)), n);
  assert(result.base().base() == expected);

  if constexpr (std::random_access_iterator<I>) {
    assert(result.stride_count() <= 1);
    // we can't say anything about the stride displacement, cause we could be using -= or +=.
  } else {
    auto const distance = n < 0 ? -n : n;
    assert(result.stride_count() == distance);
    assert(result.stride_displacement() == -n);
  }
}

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  check(bidirectional_iterator(&range[8]), 6, &range[2]);
  check(random_access_iterator(&range[7]), 4, &range[3]);
  check(contiguous_iterator(&range[5]), 5, &range[0]);

  check(bidirectional_iterator(&range[2]), 0, &range[2]);
  check(random_access_iterator(&range[3]), 0, &range[3]);
  check(contiguous_iterator(&range[0]), 0, &range[0]);

  check(bidirectional_iterator(&range[3]), -5, &range[8]);
  check(random_access_iterator(&range[3]), -3, &range[6]);
  check(contiguous_iterator(&range[3]), -1, &range[4]);
  return true;
}

int main(int, char**) {
  static_assert(test());
  test();
  return 0;
}
