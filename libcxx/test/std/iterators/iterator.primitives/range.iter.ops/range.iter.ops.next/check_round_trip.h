//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBCXX_TEST_CHECK_ROUND_TRIP_H
#define LIBCXX_TEST_CHECK_ROUND_TRIP_H

#include "test_iterators.h"

template <std::input_or_output_iterator I>
constexpr void check_round_trip(stride_counting_iterator<I> const& i, std::ptrdiff_t const n) {
  auto const distance = n < 0 ? -n : n;
  assert(i.stride_count() == distance);
  assert(i.stride_displacement() == n);
}

template <std::random_access_iterator I>
constexpr void check_round_trip(stride_counting_iterator<I> const& i, std::ptrdiff_t const n) {
  assert(i.stride_count() <= 1);
  assert(i.stride_displacement() == n < 0 ? -1 : 1);
}

template <std::input_or_output_iterator I>
constexpr bool operator==(output_iterator<I> const& x, output_iterator<I> const& y) {
  return x.base() == y.base();
}

#endif // LIBCXX_TEST_CHECK_ROUND_TRIP_H
