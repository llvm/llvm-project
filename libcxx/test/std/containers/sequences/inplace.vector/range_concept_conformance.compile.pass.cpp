//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector

#include <inplace_vector>

#include <concepts>
#include <ranges>
#include "MoveOnly.h"

template <typename T, std::size_t N>
void test() {
  using range = std::inplace_vector<T, N>;

  static_assert(std::same_as<std::ranges::iterator_t<range>, typename range::iterator>);
  static_assert(std::ranges::common_range<range>);
  static_assert(std::ranges::random_access_range<range>);
  static_assert(std::ranges::contiguous_range<range>);
  static_assert(!std::ranges::view<range>);
  static_assert(std::ranges::sized_range<range>);
  static_assert(!std::ranges::borrowed_range<range>);
  static_assert(std::ranges::viewable_range<range>);

  static_assert(std::same_as<std::ranges::iterator_t<range const>, typename range::const_iterator>);
  static_assert(std::ranges::common_range<range const>);
  static_assert(std::ranges::random_access_range<range const>);
  static_assert(std::ranges::contiguous_range<range const>);
  static_assert(!std::ranges::view<range const>);
  static_assert(std::ranges::sized_range<range const>);
  static_assert(!std::ranges::borrowed_range<range const>);
  static_assert(!std::ranges::viewable_range<range const>);
}

void tests() {
  test<int, 0>();
  test<int, 10>();
  test<MoveOnly, 0>();
  test<MoveOnly, 10>();
}
