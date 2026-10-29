//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr bool operator<(const iterator& x, const iterator& y)
// friend constexpr bool operator>(const iterator& x, const iterator& y)
// friend constexpr bool operator<=(const iterator& x, const iterator& y)
// friend constexpr bool operator>=(const iterator& x, const iterator& y)
// friend constexpr auto operator<=>(const iterator& x, const iterator& y)

#include <cassert>
#include <compare>
#include <functional>
#include <ranges>
#include <type_traits>

#include "../types.h"
#include "test_iterators.h"

// Input view: no relational comparisons.
using InputView       = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideInputIter = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;
static_assert(!std::is_invocable_v<std::less<>, StrideInputIter, StrideInputIter>);
static_assert(!std::is_invocable_v<std::greater<>, StrideInputIter, StrideInputIter>);
static_assert(!std::is_invocable_v<std::less_equal<>, StrideInputIter, StrideInputIter>);
static_assert(!std::is_invocable_v<std::greater_equal<>, StrideInputIter, StrideInputIter>);

// Forward view: no relational comparisons.
using FwdView       = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideFwdIter = std::ranges::iterator_t<std::ranges::stride_view<FwdView>>;
static_assert(!std::is_invocable_v<std::less<>, StrideFwdIter, StrideFwdIter>);
static_assert(!std::is_invocable_v<std::greater<>, StrideFwdIter, StrideFwdIter>);
static_assert(!std::is_invocable_v<std::less_equal<>, StrideFwdIter, StrideFwdIter>);
static_assert(!std::is_invocable_v<std::greater_equal<>, StrideFwdIter, StrideFwdIter>);

// Bidirectional view: no relational comparisons.
using BidirView       = BasicTestView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
using StrideBidirIter = std::ranges::iterator_t<std::ranges::stride_view<BidirView>>;
static_assert(!std::is_invocable_v<std::less<>, StrideBidirIter, StrideBidirIter>);
static_assert(!std::is_invocable_v<std::greater<>, StrideBidirIter, StrideBidirIter>);
static_assert(!std::is_invocable_v<std::less_equal<>, StrideBidirIter, StrideBidirIter>);
static_assert(!std::is_invocable_v<std::greater_equal<>, StrideBidirIter, StrideBidirIter>);

// Random access view: all relational comparisons available.
using RAView       = BasicTestView<random_access_iterator<int*>>;
using StrideRAIter = std::ranges::iterator_t<std::ranges::stride_view<RAView>>;
static_assert(std::is_invocable_v<std::less<>, StrideRAIter, StrideRAIter>);
static_assert(std::is_invocable_v<std::greater<>, StrideRAIter, StrideRAIter>);
static_assert(std::is_invocable_v<std::less_equal<>, StrideRAIter, StrideRAIter>);
static_assert(std::is_invocable_v<std::greater_equal<>, StrideRAIter, StrideRAIter>);

// three_way_comparable when the base iterator is three_way_comparable.
using ThreeWayView       = BasicTestView<three_way_contiguous_iterator<int*>>;
using StrideThreeWayIter = std::ranges::iterator_t<std::ranges::stride_view<ThreeWayView>>;
static_assert(std::three_way_comparable<StrideThreeWayIter>);

// Not three_way_comparable when the base is not.
using EqualOnlyView       = BasicTestView<cpp17_input_iterator<int*>>;
using StrideEqualOnlyIter = std::ranges::iterator_t<std::ranges::stride_view<EqualOnlyView>>;
static_assert(!std::three_way_comparable<StrideEqualOnlyIter>);

constexpr bool test() {
  {
    // <, >, <=, >= on random access range.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 9), 3);

    auto a = sv.begin(); // index 0
    auto b = sv.begin();
    ++b; // index 3
    auto c = sv.begin();
    ++c;
    ++c; // index 6

    assert(a < b);
    assert(b < c);
    assert(!(b < a));
    assert(!(a < a));

    assert(b > a);
    assert(c > b);
    assert(!(a > b));
    assert(!(a > a));

    assert(a <= b);
    assert(a <= a);
    assert(!(b <= a));

    assert(b >= a);
    assert(a >= a);
    assert(!(a >= b));
  }
  {
    // <=> on three_way_comparable base.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7};
    using Base = BasicTestView<three_way_contiguous_iterator<int*>, three_way_contiguous_iterator<int*>>;
    auto sv =
        std::ranges::stride_view(Base(three_way_contiguous_iterator(arr), three_way_contiguous_iterator(arr + 7)), 2);

    auto a = sv.begin();
    auto b = sv.begin();
    ++b;

    assert((a <=> b) == std::strong_ordering::less);
    assert((b <=> a) == std::strong_ordering::greater);
    assert((a <=> a) == std::strong_ordering::equal);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
