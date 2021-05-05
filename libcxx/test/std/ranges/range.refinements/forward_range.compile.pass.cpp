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

// std::ranges::forward_range;

#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

namespace stdr = std::ranges;

template <template <class...> class I>
constexpr bool check_forward_range() {
  constexpr bool result = stdr::forward_range<test_range<I> >;
  static_assert(stdr::forward_range<test_range<I> const> == result);
  static_assert(stdr::forward_range<test_non_const_common_range<I> > == result);
  static_assert(stdr::forward_range<test_non_const_range<I> > == result);
  static_assert(stdr::forward_range<test_common_range<I> > == result);
  static_assert(stdr::forward_range<test_common_range<I> const> == result);
  static_assert(!stdr::forward_range<test_non_const_common_range<I> const>);
  static_assert(!stdr::forward_range<test_non_const_range<I> const>);
  return result;
}

static_assert(!check_forward_range<cpp17_input_iterator>());
static_assert(!check_forward_range<cpp20_input_iterator>());
static_assert(check_forward_range<forward_iterator>());
static_assert(check_forward_range<bidirectional_iterator>());
static_assert(check_forward_range<random_access_iterator>());
static_assert(check_forward_range<contiguous_iterator>());
