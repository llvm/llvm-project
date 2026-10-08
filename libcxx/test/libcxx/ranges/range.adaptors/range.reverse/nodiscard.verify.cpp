//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <utility>

#include "test_iterators.h"

void test() {
  int range[] = {19, 28, 29, 49, 82, 94};
  auto v      = std::views::reverse(range);

  // [range.reverse.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(v).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).size();

  // [range.reverse.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::reverse(std::views::reverse(range));

  struct BidirRange : std::ranges::view_base {
    int* begin_;
    int* end_;

    constexpr BidirRange(int* b, int* e) : begin_(b), end_(e) {}

    constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{begin_}; }
    constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{begin_}; }
    constexpr bidirectional_iterator<int*> end() { return bidirectional_iterator<int*>{end_}; }
    constexpr bidirectional_iterator<const int*> end() const { return bidirectional_iterator<const int*>{end_}; }
  };
  static_assert(std::ranges::bidirectional_range<BidirRange>);
  static_assert(std::ranges::common_range<BidirRange>);
  static_assert(std::ranges::view<BidirRange>);
  static_assert(std::copyable<BidirRange>);

  { // views::reverse(x) is equivalent to subrange{end, begin, size} if x is a
    // sized subrange over reverse iterators
    using It       = bidirectional_iterator<int*>;
    using Subrange = std::ranges::subrange<It, It, std::ranges::subrange_kind::sized>;

    using ReverseIt       = std::reverse_iterator<It>;
    using ReverseSubrange = std::ranges::subrange<ReverseIt, ReverseIt, std::ranges::subrange_kind::sized>;

    BidirRange view(range, range + 6);
    ReverseSubrange subrange(ReverseIt(std::ranges::end(view)), ReverseIt(std::ranges::begin(view)), /* size */ 6);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::reverse(subrange);
  }
  { // views::reverse(x) is equivalent to subrange{end, begin} if x is an
    // unsized subrange over reverse iterators
    using It       = bidirectional_iterator<int*>;
    using Subrange = std::ranges::subrange<It, It, std::ranges::subrange_kind::unsized>;

    using ReverseIt       = std::reverse_iterator<It>;
    using ReverseSubrange = std::ranges::subrange<ReverseIt, ReverseIt, std::ranges::subrange_kind::unsized>;

    BidirRange view(range, range + 6);
    ReverseSubrange subrange(ReverseIt(std::ranges::end(view)), ReverseIt(std::ranges::begin(view)));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::reverse(subrange);
  }
  { // Otherwise, views::reverse(x) is equivalent to ranges::reverse_view{x}
    BidirRange view(range, range + 6);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::reverse(view);
  }
}
