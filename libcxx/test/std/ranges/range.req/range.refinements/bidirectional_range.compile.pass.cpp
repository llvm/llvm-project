//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class R>
// concept bidirectional_range;

#include <ranges>

#include "test_range.h"

template <template <class...> class I>
constexpr bool check_bidirectional_range() {
  constexpr bool result = std::ranges::bidirectional_range<test_range<I> >;
  static_assert(std::ranges::bidirectional_range<test_range<I> const> == result);
  static_assert(std::ranges::bidirectional_range<test_non_const_common_range<I> > == result);
  static_assert(std::ranges::bidirectional_range<test_non_const_range<I> > == result);
  static_assert(std::ranges::bidirectional_range<test_common_range<I> > == result);
  static_assert(std::ranges::bidirectional_range<test_common_range<I> const> == result);
  static_assert(!std::ranges::bidirectional_range<test_non_const_common_range<I> const>);
  static_assert(!std::ranges::bidirectional_range<test_non_const_range<I> const>);
  return result;
}

static_assert(!check_bidirectional_range<cpp17_input_iterator>());
static_assert(!check_bidirectional_range<cpp20_input_iterator>());
static_assert(!check_bidirectional_range<forward_iterator>());
static_assert(check_bidirectional_range<bidirectional_iterator>());
static_assert(check_bidirectional_range<random_access_iterator>());
static_assert(check_bidirectional_range<contiguous_iterator>());

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };

static_assert(!std::ranges::bidirectional_range<Holder<Incomplete>*>);
static_assert(!std::ranges::bidirectional_range<Holder<Incomplete>*&>);
static_assert(!std::ranges::bidirectional_range<Holder<Incomplete>*&&>);
static_assert(!std::ranges::bidirectional_range<Holder<Incomplete>* const>);
static_assert(!std::ranges::bidirectional_range<Holder<Incomplete>* const&>);
static_assert(!std::ranges::bidirectional_range<Holder<Incomplete>* const&&>);

static_assert( std::ranges::bidirectional_range<Holder<Incomplete>*[10]>);
static_assert( std::ranges::bidirectional_range<Holder<Incomplete>*(&)[10]>);
static_assert( std::ranges::bidirectional_range<Holder<Incomplete>*(&&)[10]>);
static_assert( std::ranges::bidirectional_range<Holder<Incomplete>* const[10]>);
static_assert( std::ranges::bidirectional_range<Holder<Incomplete>* const(&)[10]>);
static_assert( std::ranges::bidirectional_range<Holder<Incomplete>* const(&&)[10]>);
