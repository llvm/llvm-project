//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class R>
// concept random_access_range;

#include <ranges>

#include "test_range.h"
#include "test_iterators.h"

namespace ranges = std::ranges;

template <template <class...> class I>
constexpr bool check_range() {
  constexpr bool result = ranges::random_access_range<test_range<I> >;
  static_assert(ranges::random_access_range<test_range<I> const> == result);
  static_assert(ranges::random_access_range<test_non_const_common_range<I> > == result);
  static_assert(ranges::random_access_range<test_non_const_range<I> > == result);
  static_assert(ranges::random_access_range<test_common_range<I> > == result);
  static_assert(ranges::random_access_range<test_common_range<I> const> == result);
  static_assert(!ranges::random_access_range<test_non_const_common_range<I> const>);
  static_assert(!ranges::random_access_range<test_non_const_range<I> const>);
  return result;
}

static_assert(!check_range<cpp20_input_iterator>());
static_assert(!check_range<forward_iterator>());
static_assert(!check_range<bidirectional_iterator>());
static_assert(check_range<random_access_iterator>());
static_assert(check_range<contiguous_iterator>());
