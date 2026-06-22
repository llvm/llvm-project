//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

#include <inplace_vector>
#include <ranges>

template <class C>
constexpr bool test() {
  static_assert(std::ranges::contiguous_range<C>);
  static_assert(std::ranges::contiguous_range<const C>);
  static_assert(std::ranges::sized_range<C>);
  static_assert(std::ranges::sized_range<const C>);
  static_assert(std::ranges::common_range<C>);
  static_assert(std::ranges::common_range<const C>);
  static_assert(!std::ranges::view<C>);
  static_assert(!std::ranges::borrowed_range<C>);
  return true;
}

static_assert(test<std::inplace_vector<int, 0> >());
static_assert(test<std::inplace_vector<int, 8> >());
