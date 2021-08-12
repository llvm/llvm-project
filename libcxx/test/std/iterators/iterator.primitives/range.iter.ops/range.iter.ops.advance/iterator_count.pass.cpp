//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::advance(it, n)

#include <iterator>

#include <array>
#include <cassert>

#include "test_iterators.h"

using range_t = std::array<int, 10>;

template <std::input_or_output_iterator It>
constexpr void check_move_forward(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto first = stride_counting_iterator(It(range.begin()));
  std::ranges::advance(first, n);

  assert(first.base().base() == range.begin() + n);
  if constexpr (std::random_access_iterator<It>) {
    assert(first.stride_count() == 0 || first.stride_count() == 1);
    assert(first.stride_displacement() == 1);
  } else {
    assert(first.stride_count() == n);
    assert(first.stride_displacement() == n);
  }
}

template <std::bidirectional_iterator It>
constexpr void check_move_backward(std::ptrdiff_t const n) {
  auto range = range_t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto first = stride_counting_iterator(It(range.begin() + n));
  std::ranges::advance(first, -n);
  assert(first.base().base() == range.begin());

  if constexpr (std::random_access_iterator<It>) {
    assert(first.stride_count() == 0 || first.stride_count() == 1);
    assert(first.stride_displacement() == 1);
  } else {
    assert(first.stride_count() == n);
    assert(first.stride_displacement() == -n);
  }
}

constexpr bool test() {
  check_move_forward<cpp17_input_iterator<range_t::const_iterator> >(1);
  check_move_forward<cpp20_input_iterator<range_t::const_iterator> >(2);
  check_move_forward<forward_iterator<range_t::const_iterator> >(3);
  check_move_forward<bidirectional_iterator<range_t::const_iterator> >(4);
  check_move_forward<random_access_iterator<range_t::const_iterator> >(5);
  check_move_forward<contiguous_iterator<range_t::const_iterator> >(6);
  check_move_forward<output_iterator<range_t::iterator> >(7);

  check_move_backward<bidirectional_iterator<range_t::const_iterator> >(4);
  check_move_backward<random_access_iterator<range_t::const_iterator> >(5);
  check_move_backward<contiguous_iterator<range_t::const_iterator> >(6);

  // Zero should be checked for each case and each overload
  check_move_forward<cpp17_input_iterator<range_t::const_iterator> >(0);
  check_move_forward<cpp20_input_iterator<range_t::const_iterator> >(0);
  check_move_forward<forward_iterator<range_t::const_iterator> >(0);
  check_move_forward<bidirectional_iterator<range_t::const_iterator> >(0);
  check_move_forward<random_access_iterator<range_t::const_iterator> >(0);
  check_move_forward<output_iterator<range_t::iterator> >(0);
  check_move_backward<bidirectional_iterator<range_t::const_iterator> >(0);
  check_move_backward<random_access_iterator<range_t::const_iterator> >(0);

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
