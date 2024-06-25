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

#include <functional>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "../types.h"
#include "__compare/three_way_comparable.h"
#include "__concepts/equality_comparable.h"
#include "__concepts/same_as.h"
#include "__iterator/concepts.h"
#include "__iterator/default_sentinel.h"
#include "__iterator/distance.h"
#include "__ranges/access.h"
#include "__ranges/concepts.h"
#include "__ranges/stride_view.h"
#include "test_iterators.h"

template <class T>
concept CanPlusEqual = std::is_same_v<T&, decltype(std::declval<T>() += 1)> && requires(T& t) { t += 1; };
template <class T>
concept CanMinusEqual = std::is_same_v<T&, decltype(std::declval<T>() -= 1)> && requires(T& t) { t -= 1; };

template <class T>
concept CanMinus =
    // Note: Do *not* use std::iterator_traits here because T may not have
    // all the required pieces when it is not a forward_range.
    std::is_same_v<typename T::difference_type, decltype(std::declval<T>() - std::declval<T>())> &&
    requires(T& t) { t - t; };

template <class T>
concept CanSentinelMinus =
    // Note: Do *not* use std::iterator_traits here because T may not have
    // all the required pieces when it is not a forward_range.
    std::is_same_v<typename T::difference_type, decltype(std::declval<T>() - std::default_sentinel)> &&
    std::is_same_v<typename T::difference_type, decltype(std::default_sentinel - std::declval<T>())> && requires(T& t) {
      t - std::default_sentinel;
      std::default_sentinel - t;
    };

template <class T>
concept CanDifferencePlus = std::is_same_v<T, decltype(std::declval<T>() + 1)> && requires(T& t) { t + 1; };
template <class T>
concept CanDifferenceMinus = std::is_same_v<T, decltype(std::declval<T>() - 1)> && requires(T& t) { t - 1; };

template <class T>
concept CanPostDecrement = std::is_same_v<T, decltype(std::declval<T>()--)> && requires(T& t) { t--; };
template <class T>
concept CanPreDecrement = std::is_same_v<T&, decltype(--std::declval<T>())> && requires(T& t) { --t; };

template <class T>
concept CanSubscript = requires(T& t) { t[5]; };

// What operators are valid for an iterator derived from a stride view
// over an input view.(sized sentinel)
using InputView = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideViewOverInputViewIterator = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;

static_assert(std::weakly_incrementable<StrideViewOverInputViewIterator>);

static_assert(!CanPostDecrement<StrideViewOverInputViewIterator>);
static_assert(!CanPreDecrement<StrideViewOverInputViewIterator>);
static_assert(!CanPlusEqual<StrideViewOverInputViewIterator>);
static_assert(!CanMinusEqual<StrideViewOverInputViewIterator>);
static_assert(!CanMinus<StrideViewOverInputViewIterator>);
static_assert(!CanDifferencePlus<StrideViewOverInputViewIterator>);
static_assert(!CanDifferenceMinus<StrideViewOverInputViewIterator>);
static_assert(CanSentinelMinus<StrideViewOverInputViewIterator>);
static_assert(std::invocable<std::equal_to<>, StrideViewOverInputViewIterator, StrideViewOverInputViewIterator>);
static_assert(std::invocable<std::equal_to<>, StrideViewOverInputViewIterator, std::default_sentinel_t>);
static_assert(std::invocable<std::equal_to<>, std::default_sentinel_t, StrideViewOverInputViewIterator>);

static_assert(!std::is_invocable_v<std::less<>, StrideViewOverInputViewIterator, StrideViewOverInputViewIterator>);
static_assert(
    !std::is_invocable_v<std::less_equal<>, StrideViewOverInputViewIterator, StrideViewOverInputViewIterator>);
static_assert(!std::is_invocable_v<std::greater<>, StrideViewOverInputViewIterator, StrideViewOverInputViewIterator>);
static_assert(
    !std::is_invocable_v<std::greater_equal<>, StrideViewOverInputViewIterator, StrideViewOverInputViewIterator>);

static_assert(!CanSubscript<StrideViewOverInputViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a forward view.(sized sentinel)
using ForwardView                       = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideViewOverForwardViewIterator = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;

static_assert(std::weakly_incrementable<StrideViewOverForwardViewIterator>);

static_assert(!CanPostDecrement<StrideViewOverForwardViewIterator>);
static_assert(!CanPreDecrement<StrideViewOverForwardViewIterator>);
static_assert(!CanPlusEqual<StrideViewOverForwardViewIterator>);
static_assert(!CanMinusEqual<StrideViewOverForwardViewIterator>);
static_assert(!CanMinus<StrideViewOverForwardViewIterator>);
static_assert(!CanDifferencePlus<StrideViewOverForwardViewIterator>);
static_assert(!CanDifferenceMinus<StrideViewOverForwardViewIterator>);
static_assert(CanSentinelMinus<StrideViewOverForwardViewIterator>);
static_assert(std::invocable<std::equal_to<>, StrideViewOverForwardViewIterator, StrideViewOverForwardViewIterator>);
static_assert(std::invocable<std::equal_to<>, StrideViewOverForwardViewIterator, std::default_sentinel_t>);
static_assert(std::invocable<std::equal_to<>, std::default_sentinel_t, StrideViewOverForwardViewIterator>);

static_assert(!std::is_invocable_v<std::less<>, StrideViewOverForwardViewIterator, StrideViewOverForwardViewIterator>);
static_assert(
    !std::is_invocable_v<std::less_equal<>, StrideViewOverForwardViewIterator, StrideViewOverForwardViewIterator>);
static_assert(
    !std::is_invocable_v<std::greater<>, StrideViewOverForwardViewIterator, StrideViewOverForwardViewIterator>);
static_assert(
    !std::is_invocable_v<std::greater_equal<>, StrideViewOverForwardViewIterator, StrideViewOverForwardViewIterator>);

static_assert(!CanSubscript<StrideViewOverForwardViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a bidirectional view. (sized sentinel)
using BidirectionalView = BasicTestView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
using StrideViewOverBidirectionalViewIterator = std::ranges::iterator_t<std::ranges::stride_view<BidirectionalView>>;

static_assert(CanPostDecrement<StrideViewOverBidirectionalViewIterator>);
static_assert(CanPreDecrement<StrideViewOverBidirectionalViewIterator>);
static_assert(!CanPlusEqual<StrideViewOverBidirectionalViewIterator>);
static_assert(!CanMinusEqual<StrideViewOverBidirectionalViewIterator>);
static_assert(!CanMinus<StrideViewOverBidirectionalViewIterator>);
static_assert(!CanDifferencePlus<StrideViewOverBidirectionalViewIterator>);
static_assert(!CanDifferenceMinus<StrideViewOverBidirectionalViewIterator>);
static_assert(CanSentinelMinus<StrideViewOverBidirectionalViewIterator>);
static_assert(
    std::invocable<std::equal_to<>, StrideViewOverBidirectionalViewIterator, StrideViewOverBidirectionalViewIterator>);
static_assert(std::invocable<std::equal_to<>, StrideViewOverBidirectionalViewIterator, std::default_sentinel_t>);
static_assert(std::invocable<std::equal_to<>, std::default_sentinel_t, StrideViewOverBidirectionalViewIterator>);

static_assert(!std::is_invocable_v<std::less<>,
                                   StrideViewOverBidirectionalViewIterator,
                                   StrideViewOverBidirectionalViewIterator>);
static_assert(!std::is_invocable_v<std::less_equal<>,
                                   StrideViewOverBidirectionalViewIterator,
                                   StrideViewOverBidirectionalViewIterator>);
static_assert(!std::is_invocable_v<std::greater<>,
                                   StrideViewOverBidirectionalViewIterator,
                                   StrideViewOverBidirectionalViewIterator>);
static_assert(!std::is_invocable_v<std::greater_equal<>,
                                   StrideViewOverBidirectionalViewIterator,
                                   StrideViewOverBidirectionalViewIterator>);

static_assert(!CanSubscript<StrideViewOverBidirectionalViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a random access view. (non sized sentinel)
using RandomAccessView                       = BasicTestView<random_access_iterator<int*>>;
using StrideViewOverRandomAccessViewIterator = std::ranges::iterator_t<std::ranges::stride_view<RandomAccessView>>;

static_assert(std::weakly_incrementable<StrideViewOverRandomAccessViewIterator>);

static_assert(CanPostDecrement<StrideViewOverRandomAccessViewIterator>);
static_assert(CanPreDecrement<StrideViewOverRandomAccessViewIterator>);
static_assert(CanPlusEqual<StrideViewOverRandomAccessViewIterator>);
static_assert(CanMinusEqual<StrideViewOverRandomAccessViewIterator>);
static_assert(CanMinus<StrideViewOverRandomAccessViewIterator>);
static_assert(CanDifferencePlus<StrideViewOverRandomAccessViewIterator>);
static_assert(CanDifferenceMinus<StrideViewOverRandomAccessViewIterator>);
static_assert(!CanSentinelMinus<StrideViewOverRandomAccessViewIterator>);
static_assert(
    std::invocable<std::equal_to<>, StrideViewOverRandomAccessViewIterator, StrideViewOverRandomAccessViewIterator>);
static_assert(std::invocable<std::equal_to<>, StrideViewOverRandomAccessViewIterator, std::default_sentinel_t>);
static_assert(std::invocable<std::equal_to<>, std::default_sentinel_t, StrideViewOverRandomAccessViewIterator>);

static_assert(
    std::is_invocable_v<std::less<>, StrideViewOverRandomAccessViewIterator, StrideViewOverRandomAccessViewIterator>);
static_assert(std::is_invocable_v<std::less_equal<>,
                                  StrideViewOverRandomAccessViewIterator,
                                  StrideViewOverRandomAccessViewIterator>);
static_assert(std::is_invocable_v<std::greater<>,
                                  StrideViewOverRandomAccessViewIterator,
                                  StrideViewOverRandomAccessViewIterator>);
static_assert(std::is_invocable_v<std::greater_equal<>,
                                  StrideViewOverRandomAccessViewIterator,
                                  StrideViewOverRandomAccessViewIterator>);

static_assert(CanSubscript<StrideViewOverRandomAccessViewIterator>);

using EqualableView               = BasicTestView<cpp17_input_iterator<int*>>;
using EqualableViewStrideView     = std::ranges::stride_view<EqualableView>;
using EqualableViewStrideViewIter = std::ranges::iterator_t<EqualableViewStrideView>;

static_assert(std::equality_comparable<std::ranges::iterator_t<EqualableView>>);
static_assert(std::equality_comparable<EqualableViewStrideViewIter>);

static_assert(!std::three_way_comparable<std::ranges::iterator_t<EqualableView>>);
static_assert(!std::ranges::random_access_range<EqualableView>);
static_assert(!std::three_way_comparable<EqualableView>);

using ThreeWayComparableView           = BasicTestView<three_way_contiguous_iterator<int*>>;
using ThreeWayComparableViewStrideView = std::ranges::stride_view<ThreeWayComparableView>;
using ThreeWayComparableStrideViewIter = std::ranges::iterator_t<ThreeWayComparableViewStrideView>;

static_assert(std::three_way_comparable<std::ranges::iterator_t<ThreeWayComparableView>>);
static_assert(std::ranges::random_access_range<ThreeWayComparableView>);
static_assert(std::three_way_comparable<ThreeWayComparableStrideViewIter>);

using UnEqualableView               = ViewOverNonCopyableIterator<cpp20_input_iterator<int*>>;
using UnEqualableViewStrideView     = std::ranges::stride_view<UnEqualableView>;
using UnEqualableViewStrideViewIter = std::ranges::iterator_t<UnEqualableViewStrideView>;

static_assert(!std::equality_comparable<std::ranges::iterator_t<UnEqualableView>>);
static_assert(!std::equality_comparable<UnEqualableViewStrideViewIter>);

static_assert(!std::three_way_comparable<std::ranges::iterator_t<UnEqualableView>>);
static_assert(!std::ranges::random_access_range<UnEqualableView>);
static_assert(!std::three_way_comparable<UnEqualableView>);

template <typename Iter>
  requires std::sized_sentinel_for<Iter, Iter> && (!std::forward_iterator<Iter>)
constexpr bool test_non_forward_operator_minus(Iter zero_begin, Iter one_begin, Iter end) {
  using Base = BasicTestView<Iter, Iter>;
  // Test the non-forward-range operator- between two iterators (i.e., ceil).
  // First, what operators are valid for an iterator derived from a stride view
  // over a sized input view.
  using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<Base>>;

  static_assert(std::weakly_incrementable<StrideViewIterator>);

  static_assert(!CanPostDecrement<StrideViewIterator>);
  static_assert(!CanPreDecrement<StrideViewIterator>);
  static_assert(!CanPlusEqual<StrideViewIterator>);
  static_assert(!CanMinusEqual<StrideViewIterator>);
  static_assert(!CanDifferencePlus<StrideViewIterator>);
  static_assert(!CanDifferenceMinus<StrideViewIterator>);
  static_assert(CanSentinelMinus<StrideViewIterator>);

  static_assert(!std::is_invocable_v<std::less<>, StrideViewIterator, StrideViewIterator>);
  static_assert(!std::is_invocable_v<std::less_equal<>, StrideViewIterator, StrideViewIterator>);
  static_assert(!std::is_invocable_v<std::greater<>, StrideViewIterator, StrideViewIterator>);
  static_assert(!std::is_invocable_v<std::greater_equal<>, StrideViewIterator, StrideViewIterator>);
  static_assert(std::is_invocable_v<std::equal_to<>, StrideViewIterator, StrideViewIterator>);
  static_assert(std::is_invocable_v<std::equal_to<>, std::default_sentinel_t, StrideViewIterator>);
  static_assert(std::is_invocable_v<std::equal_to<>, StrideViewIterator, std::default_sentinel_t>);
  static_assert(!CanSubscript<StrideViewIterator>);

  auto base_view_offset_zero             = Base(zero_begin, end);
  auto base_view_offset_one              = Base(one_begin, end);
  auto stride_view_over_base_zero_offset = std::ranges::stride_view(base_view_offset_zero, 3);
  auto stride_view_over_base_one_offset  = std::ranges::stride_view(base_view_offset_one, 3);

  auto sv_zero_offset_begin = stride_view_over_base_zero_offset.begin();
  auto sv_one_offset_begin  = stride_view_over_base_one_offset.begin();

  auto sv_zero_offset_zeroth_index = sv_zero_offset_begin;
  auto sv_zero_offset_third_index  = ++sv_zero_offset_begin;
  auto sv_zero_offset_sixth_index  = ++sv_zero_offset_begin;

  auto sv_one_offset_oneth_index  = sv_one_offset_begin;
  auto sv_one_offset_fourth_index = ++sv_one_offset_begin;

  static_assert(std::sized_sentinel_for<std::ranges::iterator_t<Base>, std::ranges::iterator_t<Base>>);
  static_assert(CanMinus<decltype(sv_zero_offset_begin)>);

  // Check positive __n with exact multiple of left's stride.
  assert(sv_zero_offset_third_index - sv_zero_offset_zeroth_index == 1);
  assert(sv_zero_offset_sixth_index - sv_zero_offset_zeroth_index == 2);
  // Check positive __n with non-exact multiple of left's stride (will do ceil here).
  assert(sv_one_offset_oneth_index - sv_zero_offset_zeroth_index == 1);
  assert(sv_one_offset_fourth_index - sv_zero_offset_zeroth_index == 2);

  // Check negative __n with exact multiple of left's stride.
  assert(sv_zero_offset_zeroth_index - sv_zero_offset_third_index == -1);
  assert(sv_zero_offset_zeroth_index - sv_zero_offset_sixth_index == -2);
  // Check negative __n with non-exact multiple of left's stride (will do ceil here).
  assert(sv_zero_offset_zeroth_index - sv_one_offset_oneth_index == -1);
  assert(sv_zero_offset_zeroth_index - sv_one_offset_fourth_index == -2);

  assert(stride_view_over_base_zero_offset.end() == std::default_sentinel);
  assert(std::default_sentinel == stride_view_over_base_zero_offset.end());

  assert(stride_view_over_base_zero_offset.end() - std::default_sentinel == 0);
  assert(std::default_sentinel - stride_view_over_base_zero_offset.begin() ==
         std::ranges::distance(stride_view_over_base_zero_offset));
  assert(stride_view_over_base_zero_offset.begin() - std::default_sentinel ==
         -std::ranges::distance(stride_view_over_base_zero_offset));

  return true;
}

template <std::forward_iterator Iter>
constexpr bool test_forward_operator_minus(Iter begin, Iter end) {
  // Test the forward-range operator- between two iterators (i.e., no ceil).
  using Base = BasicTestView<Iter, Iter>;
  //int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // First, what operators are valid for an iterator derived from a stride view
  // over a sized forward view (even though it is actually much more than that!).
  using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<Base>>;

  static_assert(std::weakly_incrementable<StrideViewIterator>);
  static_assert(CanMinus<StrideViewIterator>);
  static_assert(CanSentinelMinus<StrideViewIterator>);

  auto base_view_offset_zero             = Base(begin, end);
  auto base_view_offset_one              = Base(begin + 1, end);
  auto stride_view_over_base_zero_offset = std::ranges::stride_view(base_view_offset_zero, 3);
  auto stride_view_over_base_one_offset  = std::ranges::stride_view(base_view_offset_one, 3);

  auto sv_zero_offset_begin = stride_view_over_base_zero_offset.begin();
  auto sv_one_offset_begin  = stride_view_over_base_one_offset.begin();

  auto sv_zero_offset_should_be_one   = sv_zero_offset_begin;
  auto sv_zero_offset_should_be_four  = ++sv_zero_offset_begin;
  auto sv_zero_offset_should_be_seven = ++sv_zero_offset_begin;

  auto sv_one_offset_should_be_two  = sv_one_offset_begin;
  auto sv_one_offset_should_be_five = ++sv_one_offset_begin;

  static_assert(std::sized_sentinel_for<std::ranges::iterator_t<Base>, std::ranges::iterator_t<Base>>);
  static_assert(CanMinus<decltype(sv_zero_offset_begin)>);
  static_assert(std::forward_iterator<std::ranges::iterator_t<Base>>);

  // Check positive __n with exact multiple of left's stride.
  assert(sv_zero_offset_should_be_four - sv_zero_offset_should_be_one == 1);
  assert(sv_zero_offset_should_be_seven - sv_zero_offset_should_be_one == 2);

  // Check positive __n with non-exact multiple of left's stride.
  assert(sv_one_offset_should_be_two - sv_zero_offset_should_be_one == 0);
  assert(sv_one_offset_should_be_five - sv_zero_offset_should_be_one == 1);

  // Check negative __n with exact multiple of left's stride.
  assert(sv_zero_offset_should_be_one - sv_zero_offset_should_be_four == -1);
  assert(sv_zero_offset_should_be_one - sv_zero_offset_should_be_seven == -2);

  // Check negative __n with non-exact multiple of left's stride.
  assert(sv_zero_offset_should_be_one - sv_one_offset_should_be_two == 0);
  assert(sv_zero_offset_should_be_one - sv_one_offset_should_be_five == -1);

  // Make sure that all sentinel operations work!
  assert(stride_view_over_base_zero_offset.end() == std::default_sentinel);
  assert(std::default_sentinel == stride_view_over_base_zero_offset.end());

  assert(stride_view_over_base_zero_offset.end() - std::default_sentinel == 0);
  assert(std::default_sentinel - stride_view_over_base_zero_offset.begin() ==
         std::ranges::distance(stride_view_over_base_zero_offset));
  assert(stride_view_over_base_zero_offset.begin() - std::default_sentinel ==
         -std::ranges::distance(stride_view_over_base_zero_offset));
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
  // another __missing_ != 0 value.
  strider_iter++;
  assert(strider_iter == strider.end());

  // Make sure that all sentinel operations work!
  assert(strider.end() == std::default_sentinel);
  assert(std::default_sentinel == strider.end());

  assert(strider_iter - std::default_sentinel == 0);
  assert(std::default_sentinel - strider.end() == 0);
  assert(std::default_sentinel - strider_iter == 0);

  // Let's make sure that the newly regenerated __missing_ gets used.
  strider_iter += -2;
  assert(*strider_iter == 1);

  return true;
}

int main(int, char**) {
  {
    constexpr int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    test_forward_operator_minus(arr, arr + 10);
    test_forward_operator_minus(vec.begin(), vec.end());
  }

  {
    int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    test_non_forward_operator_minus(SizedInputIterator(arr), SizedInputIterator(arr + 1), SizedInputIterator(arr + 10));
  }

  test_properly_handling_missing();
  static_assert(test_properly_handling_missing());
  return 0;
}
