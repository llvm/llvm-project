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
concept can_calculate_distance_between_non_sentinel = requires(T& t) { t - t; };

constexpr bool operator_tests() {
  {
    // What operators are valid for an iterator derived from a stride view
    // over an input view.
    using View               = InputView<cpp17_input_iterator<int*>>;
    using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<View>>;

    static_assert(is_plus_plusable_post<StrideViewIterator>);
    static_assert(is_plus_plusable_pre<StrideViewIterator>);
    static_assert(!is_minus_minusable_post<StrideViewIterator>);
    static_assert(!is_minus_minusable_pre<StrideViewIterator>);
    static_assert(!is_plus_equalable<StrideViewIterator>);
    static_assert(!is_minus_equalable<StrideViewIterator>);
    static_assert(!is_iterator_minusable<StrideViewIterator>);
    static_assert(!is_difference_plusable<StrideViewIterator>);
    static_assert(!is_difference_minusable<StrideViewIterator>);
    static_assert(!is_relationally_comparable<StrideViewIterator>);
  }
  {
    // What operators are valid for an iterator derived from a stride view
    // over a forward view.
    using View               = InputView<forward_iterator<int*>>;
    using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<View>>;

    static_assert(is_plus_plusable_post<StrideViewIterator>);
    static_assert(is_plus_plusable_pre<StrideViewIterator>);
    static_assert(!is_minus_minusable_post<StrideViewIterator>);
    static_assert(!is_minus_minusable_pre<StrideViewIterator>);
    static_assert(!is_plus_equalable<StrideViewIterator>);
    static_assert(!is_minus_equalable<StrideViewIterator>);
    static_assert(!is_iterator_minusable<StrideViewIterator>);
    static_assert(!is_difference_plusable<StrideViewIterator>);
    static_assert(!is_difference_minusable<StrideViewIterator>);
    static_assert(!is_relationally_comparable<StrideViewIterator>);
  }
  {
    // What operators are valid for an iterator derived from a stride view
    // over a sized input view.
    using View               = InputView<SizedInputIterator>;
    using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<View>>;

    static_assert(is_plus_plusable_post<StrideViewIterator>);
    static_assert(is_plus_plusable_pre<StrideViewIterator>);
    static_assert(!is_minus_minusable_post<StrideViewIterator>);
    static_assert(!is_minus_minusable_pre<StrideViewIterator>);
    static_assert(!is_plus_equalable<StrideViewIterator>);
    static_assert(!is_minus_equalable<StrideViewIterator>);
    static_assert(is_iterator_minusable<StrideViewIterator>);
    static_assert(!is_difference_plusable<StrideViewIterator>);
    static_assert(!is_difference_minusable<StrideViewIterator>);
    static_assert(!is_relationally_comparable<StrideViewIterator>);
  }
  {
    // What operators are valid for an iterator derived from a stride view
    // over a sized forward view.
    using View               = InputView<SizedForwardIterator>;
    using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<View>>;

    static_assert(is_plus_plusable_post<StrideViewIterator>);
    static_assert(is_plus_plusable_pre<StrideViewIterator>);
    static_assert(!is_minus_minusable_post<StrideViewIterator>);
    static_assert(!is_minus_minusable_pre<StrideViewIterator>);
    static_assert(!is_plus_equalable<StrideViewIterator>);
    static_assert(!is_minus_equalable<StrideViewIterator>);
    static_assert(is_iterator_minusable<StrideViewIterator>);
    static_assert(!is_difference_plusable<StrideViewIterator>);
    static_assert(!is_difference_minusable<StrideViewIterator>);
    static_assert(!is_relationally_comparable<StrideViewIterator>);
  }
  {
    // What operators are valid for an iterator derived from a stride view
    // over a bidirectional view.
    using View               = InputView<bidirectional_iterator<int*>>;
    using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<View>>;

    static_assert(is_plus_plusable_post<StrideViewIterator>);
    static_assert(is_plus_plusable_pre<StrideViewIterator>);
    static_assert(is_minus_minusable_post<StrideViewIterator>);
    static_assert(is_minus_minusable_pre<StrideViewIterator>);
    static_assert(!is_plus_equalable<StrideViewIterator>);
    static_assert(!is_minus_equalable<StrideViewIterator>);
    static_assert(!is_iterator_minusable<StrideViewIterator>);
    static_assert(!is_difference_plusable<StrideViewIterator>);
    static_assert(!is_difference_minusable<StrideViewIterator>);
    static_assert(!is_relationally_comparable<StrideViewIterator>);
  }
  {
    // What operators are valid for an iterator derived from a stride view
    // over a random access view.
    using View               = InputView<random_access_iterator<int*>>;
    using StrideViewIterator = std::ranges::iterator_t<std::ranges::stride_view<View>>;

    static_assert(is_plus_plusable_post<StrideViewIterator>);
    static_assert(is_plus_plusable_pre<StrideViewIterator>);
    static_assert(is_minus_minusable_post<StrideViewIterator>);
    static_assert(is_minus_minusable_pre<StrideViewIterator>);
    static_assert(is_plus_equalable<StrideViewIterator>);
    static_assert(is_minus_equalable<StrideViewIterator>);
    static_assert(is_iterator_minusable<StrideViewIterator>);
    static_assert(is_difference_plusable<StrideViewIterator>);
    static_assert(is_difference_minusable<StrideViewIterator>);
    static_assert(is_relationally_comparable<StrideViewIterator>);
  }
  {
    using Base = InputView<SizedForwardIterator, SizedForwardIterator>;
    // Test the forward-range operator- between two iterators (i.e., no ceil).
    int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto rav_zero    = Base(SizedForwardIterator(arr), SizedForwardIterator(arr + 10));
    auto rav_one     = Base(SizedForwardIterator(arr + 1), SizedForwardIterator(arr + 10));
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
    static_assert(std::forward_iterator<SizedForwardIterator>);

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
  }

  {
    using Base = InputView<SizedInputIterator, SizedInputIterator>;
    // Test the non-forward-range operator- between two iterators (i.e., ceil).
    int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
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
  }
  {
    // Check whether __missing_ gets handled properly.
    using Base = SimpleView;
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
  }
  {
    using EqualableView = InputView<cpp17_input_iterator<int*>>;
    using Stride        = std::ranges::stride_view<EqualableView>;
    using StrideIter    = std::ranges::iterator_t<Stride>;

    static_assert(is_relationally_equalable<std::ranges::iterator_t<EqualableView>>);
    static_assert(is_relationally_equalable<StrideIter>);

    static_assert(!std::three_way_comparable<std::ranges::iterator_t<EqualableView>>);
    static_assert(!std::ranges::random_access_range<EqualableView>);
    static_assert(!is_three_way_comparable<EqualableView>);
  }
  {
    using ThreeWayComparableView = InputView<rvalue_iterator<int*>>;
    using Stride                 = std::ranges::stride_view<ThreeWayComparableView>;
    using StrideIter             = std::ranges::iterator_t<Stride>;

    static_assert(std::three_way_comparable<std::ranges::iterator_t<ThreeWayComparableView>>);
    static_assert(std::ranges::random_access_range<ThreeWayComparableView>);
    static_assert(is_three_way_comparable<StrideIter>);
  }
  {
    using UnEqualableView =
        ViewOverNonCopyable<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>;
    using Stride     = std::ranges::stride_view<UnEqualableView>;
    using StrideIter = std::ranges::iterator_t<Stride>;

    static_assert(!is_relationally_equalable<std::ranges::iterator_t<UnEqualableView>>);
    static_assert(!is_relationally_equalable<StrideIter>);

    static_assert(!std::three_way_comparable<std::ranges::iterator_t<UnEqualableView>>);
    static_assert(!std::ranges::random_access_range<UnEqualableView>);
    static_assert(!is_three_way_comparable<UnEqualableView>);
  }

  return true;
}

int main(int, char**) {
  operator_tests();
  static_assert(operator_tests());

  return 0;
}
