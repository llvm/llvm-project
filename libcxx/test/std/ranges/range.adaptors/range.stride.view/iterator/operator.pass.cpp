//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr __iterator& operator++()
// constexpr void operator++(int)
// constexpr __iterator operator++(int)
// constexpr __iterator& operator--()
// constexpr __iterator operator--(int)
// constexpr __iterator& operator+=(difference_type __n)
// constexpr __iterator& operator-=(difference_type __n)
// friend constexpr bool operator==(__iterator const& __x, default_sentinel_t)
// friend constexpr bool operator==(__iterator const& __x, __iterator const& __y)
// friend constexpr bool operator<(__iterator const& __x, __iterator const& __y)
// friend constexpr bool operator>(__iterator const& __x, __iterator const& __y)
// friend constexpr bool operator<=(__iterator const& __x, __iterator const& __y)
// friend constexpr bool operator>=(__iterator const& __x, __iterator const& __y)
// friend constexpr bool operator<=>(__iterator const& __x, __iterator const& __y)

#include <iterator>
#include <ranges>
#include <type_traits>

#include "../types.h"
#include "__ranges/concepts.h"
#include "__ranges/stride_view.h"
#include "test_iterators.h"

template <class T>
concept is_plus_equalable = requires(T& t) { t += 1; };
template <class T>
concept is_minus_equalable = requires(T& t) { t -= 1; };

template <class T>
concept is_iterator_minusable = requires(T& t) { t - t; };
template <class T>
concept is_difference_plusable = requires(T& t) { t + 1; };
template <class T>
concept is_difference_minusable = requires(T& t) { t - 1; };

template <class T>
concept is_relationally_comparable = requires(T& t) {
  t < t;
  t > t;
  t <= t;
  t >= t;
};

template <class T>
concept is_relationally_equalable = requires(T& t) { t == t; };

template <class T>
concept is_three_way_comparable = requires(T& t) { t <=> t; };

template <class T>
concept is_plus_plusable_post = requires(T& t) { t++; };
template <class T>
concept is_plus_plusable_pre = requires(T& t) { ++t; };
template <class T>
concept is_minus_minusable_post = requires(T& t) { t--; };
template <class T>
concept is_minus_minusable_pre = requires(T& t) { --t; };

template <class T>
concept is_indexable = requires(T& t) { t[5]; };

template <class T>
concept can_calculate_distance_between_non_sentinel = requires(T& t) { t - t; };

// What operators are valid for an iterator derived from a stride view
// over an input view.
using StrideViewOverInputViewIterator =
    std::ranges::iterator_t<std::ranges::stride_view<BasicTestView<cpp17_input_iterator<int*>>>>;

static_assert(is_plus_plusable_post<StrideViewOverInputViewIterator>);
static_assert(is_plus_plusable_pre<StrideViewOverInputViewIterator>);
static_assert(!is_minus_minusable_post<StrideViewOverInputViewIterator>);
static_assert(!is_minus_minusable_pre<StrideViewOverInputViewIterator>);
static_assert(!is_plus_equalable<StrideViewOverInputViewIterator>);
static_assert(!is_minus_equalable<StrideViewOverInputViewIterator>);
static_assert(!is_iterator_minusable<StrideViewOverInputViewIterator>);
static_assert(!is_difference_plusable<StrideViewOverInputViewIterator>);
static_assert(!is_difference_minusable<StrideViewOverInputViewIterator>);
static_assert(!is_relationally_comparable<StrideViewOverInputViewIterator>);
static_assert(!is_indexable<StrideViewOverInputViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a forward view.
using ForwardView                       = BasicTestView<forward_iterator<int*>>;
using StrideViewOverForwardViewIterator = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;

static_assert(is_plus_plusable_post<StrideViewOverForwardViewIterator>);
static_assert(is_plus_plusable_pre<StrideViewOverForwardViewIterator>);
static_assert(!is_minus_minusable_post<StrideViewOverForwardViewIterator>);
static_assert(!is_minus_minusable_pre<StrideViewOverForwardViewIterator>);
static_assert(!is_plus_equalable<StrideViewOverForwardViewIterator>);
static_assert(!is_minus_equalable<StrideViewOverForwardViewIterator>);
static_assert(!is_iterator_minusable<StrideViewOverForwardViewIterator>);
static_assert(!is_difference_plusable<StrideViewOverForwardViewIterator>);
static_assert(!is_difference_minusable<StrideViewOverForwardViewIterator>);
static_assert(!is_relationally_comparable<StrideViewOverForwardViewIterator>);
static_assert(!is_indexable<StrideViewOverForwardViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a bidirectional view.
using BidirectionalView                       = BasicTestView<bidirectional_iterator<int*>>;
using StrideViewOverBidirectionalViewIterator = std::ranges::iterator_t<std::ranges::stride_view<BidirectionalView>>;

static_assert(is_plus_plusable_post<StrideViewOverBidirectionalViewIterator>);
static_assert(is_plus_plusable_pre<StrideViewOverBidirectionalViewIterator>);
static_assert(is_minus_minusable_post<StrideViewOverBidirectionalViewIterator>);
static_assert(is_minus_minusable_pre<StrideViewOverBidirectionalViewIterator>);
static_assert(!is_plus_equalable<StrideViewOverBidirectionalViewIterator>);
static_assert(!is_minus_equalable<StrideViewOverBidirectionalViewIterator>);
static_assert(!is_iterator_minusable<StrideViewOverBidirectionalViewIterator>);
static_assert(!is_difference_plusable<StrideViewOverBidirectionalViewIterator>);
static_assert(!is_difference_minusable<StrideViewOverBidirectionalViewIterator>);
static_assert(!is_relationally_comparable<StrideViewOverBidirectionalViewIterator>);
static_assert(!is_indexable<StrideViewOverBidirectionalViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a random access view.
using RandomAccessView                       = BasicTestView<random_access_iterator<int*>>;
using StrideViewOverRandomAccessViewIterator = std::ranges::iterator_t<std::ranges::stride_view<RandomAccessView>>;

static_assert(is_plus_plusable_post<StrideViewOverRandomAccessViewIterator>);
static_assert(is_plus_plusable_pre<StrideViewOverRandomAccessViewIterator>);
static_assert(is_minus_minusable_post<StrideViewOverRandomAccessViewIterator>);
static_assert(is_minus_minusable_pre<StrideViewOverRandomAccessViewIterator>);
static_assert(is_plus_equalable<StrideViewOverRandomAccessViewIterator>);
static_assert(is_minus_equalable<StrideViewOverRandomAccessViewIterator>);
static_assert(is_iterator_minusable<StrideViewOverRandomAccessViewIterator>);
static_assert(is_difference_plusable<StrideViewOverRandomAccessViewIterator>);
static_assert(is_difference_minusable<StrideViewOverRandomAccessViewIterator>);
static_assert(is_relationally_comparable<StrideViewOverRandomAccessViewIterator>);
static_assert(is_indexable<StrideViewOverRandomAccessViewIterator>);

using EqualableView               = BasicTestView<cpp17_input_iterator<int*>>;
using EqualableViewStrideView     = std::ranges::stride_view<EqualableView>;
using EqualableViewStrideViewIter = std::ranges::iterator_t<EqualableViewStrideView>;

static_assert(is_relationally_equalable<std::ranges::iterator_t<EqualableView>>);
static_assert(is_relationally_equalable<EqualableViewStrideViewIter>);

static_assert(!std::three_way_comparable<std::ranges::iterator_t<EqualableView>>);
static_assert(!std::ranges::random_access_range<EqualableView>);
static_assert(!is_three_way_comparable<EqualableView>);

using ThreeWayComparableView           = BasicTestView<rvalue_iterator<int*>>;
using ThreeWayComparableViewStrideView = std::ranges::stride_view<ThreeWayComparableView>;
using ThreeWayComparableStrideViewIter = std::ranges::iterator_t<ThreeWayComparableViewStrideView>;

static_assert(std::three_way_comparable<std::ranges::iterator_t<ThreeWayComparableView>>);
static_assert(std::ranges::random_access_range<ThreeWayComparableView>);
static_assert(is_three_way_comparable<ThreeWayComparableStrideViewIter>);

using UnEqualableView               = ViewOverNonCopyableIterator<cpp20_input_iterator<int*>>;
using UnEqualableViewStrideView     = std::ranges::stride_view<UnEqualableView>;
using UnEqualableViewStrideViewIter = std::ranges::iterator_t<UnEqualableViewStrideView>;

static_assert(!is_relationally_equalable<std::ranges::iterator_t<UnEqualableView>>);
static_assert(!is_relationally_equalable<UnEqualableViewStrideViewIter>);

static_assert(!std::three_way_comparable<std::ranges::iterator_t<UnEqualableView>>);
static_assert(!std::ranges::random_access_range<UnEqualableView>);
static_assert(!is_three_way_comparable<UnEqualableView>);

constexpr bool test_non_forward_operator_minus() {
  using Base = BasicTestView<SizedInputIterator, SizedInputIterator>;
  // Test the non-forward-range operator- between two iterators (i.e., ceil).
  int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // First, what operators are valid for an iterator derived from a stride view
  // over a sized input view.
  using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<Base>>;

  static_assert(is_plus_plusable_post<StrideViewIterator>);
  static_assert(is_plus_plusable_pre<StrideViewIterator>);
  static_assert(!is_minus_minusable_post<StrideViewIterator>);
  static_assert(!is_minus_minusable_pre<StrideViewIterator>);
  static_assert(!is_plus_equalable<StrideViewIterator>);
  static_assert(!is_minus_equalable<StrideViewIterator>);
  static_assert(!is_difference_plusable<StrideViewIterator>);
  static_assert(!is_difference_minusable<StrideViewIterator>);
  static_assert(!is_relationally_comparable<StrideViewIterator>);
  static_assert(!is_indexable<StrideViewIterator>);

  auto rav_zero    = Base(SizedInputIterator(arr), SizedInputIterator(arr + 10));
  auto rav_one     = Base(SizedInputIterator(arr + 1), SizedInputIterator(arr + 10));
  auto stride_zoff = std::ranges::stride_view(rav_zero, 3);
  auto stride_ooff = std::ranges::stride_view(rav_one, 3);

  auto stride_zoff_begin = stride_zoff.begin();
  auto stride_ooff_begin = stride_ooff.begin();

  auto stride_zoff_one   = stride_zoff_begin;
  auto stride_zoff_four  = ++stride_zoff_begin;
  auto stride_zoff_seven = ++stride_zoff_begin;

  auto stride_ooff_two  = stride_ooff_begin;
  auto stride_ooff_five = ++stride_ooff_begin;

  static_assert(std::sized_sentinel_for<std::ranges::iterator_t<Base>, std::ranges::iterator_t<Base>>);
  static_assert(can_calculate_distance_between_non_sentinel<decltype(stride_zoff_begin)>);

  assert(*stride_zoff_one == 1);
  assert(*stride_zoff_four == 4);
  assert(*stride_zoff_seven == 7);

  assert(*stride_ooff_two == 2);
  assert(*stride_ooff_five == 5);

  // Check positive __n with exact multiple of left's stride.
  assert(stride_zoff_four - stride_zoff_one == 1);
  assert(stride_zoff_seven - stride_zoff_one == 2);
  // Check positive __n with non-exact multiple of left's stride.
  assert(stride_ooff_two - stride_zoff_one == 1);
  assert(stride_ooff_five - stride_zoff_one == 2);

  // Check negative __n with exact multiple of left's stride.
  assert(stride_zoff_one - stride_zoff_four == -1);
  assert(stride_zoff_one - stride_zoff_seven == -2);
  // Check negative __n with non-exact multiple of left's stride.
  assert(stride_zoff_one - stride_ooff_two == -1);
  assert(stride_zoff_one - stride_ooff_five == -2);

  return true;
}

constexpr bool test_forward_operator_minus() {
  // Test the forward-range operator- between two iterators (i.e., no ceil).
  using Base = BasicTestView<int*, int*>;
  int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // First, what operators are valid for an iterator derived from a stride view
  // over a sized forward view (even though it is actually much more than that!).
  using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<Base>>;

  static_assert(is_plus_plusable_post<StrideViewIterator>);
  static_assert(is_plus_plusable_pre<StrideViewIterator>);
  static_assert(is_iterator_minusable<StrideViewIterator>);

  auto rav_zero    = Base(arr, arr + 10);
  auto rav_one     = Base(arr + 1, arr + 10);
  auto stride_zoff = std::ranges::stride_view(rav_zero, 3);
  auto stride_ooff = std::ranges::stride_view(rav_one, 3);

  auto stride_zoff_begin = stride_zoff.begin();
  auto stride_ooff_begin = stride_ooff.begin();

  auto stride_zoff_one   = stride_zoff_begin;
  auto stride_zoff_four  = ++stride_zoff_begin;
  auto stride_zoff_seven = ++stride_zoff_begin;

  auto stride_ooff_two  = stride_ooff_begin;
  auto stride_ooff_five = ++stride_ooff_begin;

  static_assert(std::sized_sentinel_for<std::ranges::iterator_t<Base>, std::ranges::iterator_t<Base>>);
  static_assert(can_calculate_distance_between_non_sentinel<decltype(stride_zoff_begin)>);
  static_assert(std::forward_iterator<std::ranges::iterator_t<Base>>);

  assert(*stride_zoff_one == 1);
  assert(*stride_zoff_four == 4);
  assert(*stride_zoff_seven == 7);

  assert(*stride_ooff_two == 2);
  assert(*stride_ooff_five == 5);
  // Check positive __n with exact multiple of left's stride.
  assert(stride_zoff_four - stride_zoff_one == 1);
  assert(stride_zoff_seven - stride_zoff_one == 2);

  // Check positive __n with non-exact multiple of left's stride.
  assert(stride_ooff_two - stride_zoff_one == 0);
  assert(stride_ooff_five - stride_zoff_one == 1);

  // Check negative __n with exact multiple of left's stride.
  assert(stride_zoff_one - stride_zoff_four == -1);
  assert(stride_zoff_one - stride_zoff_seven == -2);

  // Check negative __n with non-exact multiple of left's stride.
  assert(stride_zoff_one - stride_ooff_two == 0);
  assert(stride_zoff_one - stride_ooff_five == -1);
  return true;
}

constexpr bool test_properly_handling_missing() {
  // Check whether __missing_ gets handled properly.
  using Base = BasicTestView<int*, int*>;
  int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto base    = Base(arr, arr + 10);
  auto strider = std::ranges::stride_view<Base>(base, 7);

  auto strider_iter = strider.end();

  strider_iter--;
  assert(*strider_iter == 8);

  // Now that we are back among the valid, we should
  // have a normal stride length back (i.e., __missing_
  // should be equal to 0).
  strider_iter--;
  assert(*strider_iter == 1);

  strider_iter++;
  assert(*strider_iter == 8);

  // By striding past the end, we are going to generate
  // another __missing_ != 0 value. Let's make sure
  // that it gets generated and used.
  strider_iter++;
  assert(strider_iter == strider.end());

  strider_iter--;
  assert(*strider_iter == 8);

  strider_iter--;
  assert(*strider_iter == 1);
  return true;
}

int main(int, char**) {
  test_forward_operator_minus();
  static_assert(test_forward_operator_minus());

  test_non_forward_operator_minus();
  static_assert(test_non_forward_operator_minus());

  test_properly_handling_missing();
  static_assert(test_properly_handling_missing());
  return 0;
}
