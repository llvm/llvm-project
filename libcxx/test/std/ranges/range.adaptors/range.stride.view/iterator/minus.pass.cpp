//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr difference_type operator-(const iterator& x, const iterator& y)
// friend constexpr difference_type operator-(default_sentinel_t, const iterator& x)
// friend constexpr difference_type operator-(const iterator& x, default_sentinel_t)
// friend constexpr iterator operator-(const iterator& i, difference_type s)

#include <cassert>
#include <ranges>
#include <type_traits>
#include <vector>

#include "../types.h"
#include "test_iterators.h"

template <class T>
concept CanMinus = std::is_same_v<typename T::difference_type, decltype(std::declval<T>() - std::declval<T>())> &&
                   requires(T& t) { t - t; };

template <class T>
concept CanSentinelMinus =
    std::is_same_v<typename T::difference_type, decltype(std::declval<T>() - std::default_sentinel)> &&
    std::is_same_v<typename T::difference_type, decltype(std::default_sentinel - std::declval<T>())> && requires(T& t) {
      t - std::default_sentinel;
      std::default_sentinel - t;
    };

template <class T>
concept CanDifferenceMinus = std::is_same_v<T, decltype(std::declval<T>() - 1)> && requires(T& t) { t - 1; };

// Input view: has sentinel minus but not iter-iter minus or difference minus.
using InputView = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideViewOverInputViewIter = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;
static_assert(!CanMinus<StrideViewOverInputViewIter>);
static_assert(!CanDifferenceMinus<StrideViewOverInputViewIter>);
static_assert(CanSentinelMinus<StrideViewOverInputViewIter>);

// Forward view: has sentinel minus but not iter-iter minus or difference minus.
using ForwardView                   = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideViewOverForwardViewIter = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;
static_assert(!CanMinus<StrideViewOverForwardViewIter>);
static_assert(!CanDifferenceMinus<StrideViewOverForwardViewIter>);
static_assert(CanSentinelMinus<StrideViewOverForwardViewIter>);

// Bidirectional view: has sentinel minus but not iter-iter minus or difference minus.
using BidirView = BasicTestView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
using StrideViewOverBidirViewIter = std::ranges::iterator_t<std::ranges::stride_view<BidirView>>;
static_assert(!CanMinus<StrideViewOverBidirViewIter>);
static_assert(!CanDifferenceMinus<StrideViewOverBidirViewIter>);
static_assert(CanSentinelMinus<StrideViewOverBidirViewIter>);

// Random access view: has iter-iter minus and difference minus, but no sentinel minus (non-sized sentinel).
using RAView                   = BasicTestView<random_access_iterator<int*>>;
using StrideViewOverRAViewIter = std::ranges::iterator_t<std::ranges::stride_view<RAView>>;
static_assert(CanMinus<StrideViewOverRAViewIter>);
static_assert(CanDifferenceMinus<StrideViewOverRAViewIter>);
static_assert(!CanSentinelMinus<StrideViewOverRAViewIter>);

template <typename Iter>
  requires std::sized_sentinel_for<Iter, Iter> && (!std::forward_iterator<Iter>)
constexpr bool test_non_forward_minus(Iter zero_begin, Iter one_begin, Iter end) {
  using Base = BasicTestView<Iter, Iter>;

  auto base_zero = Base(zero_begin, end);
  auto base_one  = Base(one_begin, end);
  auto sv_zero   = std::ranges::stride_view(base_zero, 3);
  auto sv_one    = std::ranges::stride_view(base_one, 3);

  auto begin0 = sv_zero.begin();
  auto mid0   = begin0;
  ++mid0; // stride 3 -> index 3
  auto far0 = mid0;
  ++far0; // stride 3 -> index 6

  auto begin1 = sv_one.begin();
  auto mid1   = begin1;
  ++mid1; // stride 3 -> index 4

  // Positive differences (uses ceil for non-forward).
  assert(mid0 - begin0 == 1);
  assert(far0 - begin0 == 2);
  assert(begin1 - begin0 == 1);
  assert(mid1 - begin0 == 2);

  // Negative differences.
  assert(begin0 - mid0 == -1);
  assert(begin0 - far0 == -2);
  assert(begin0 - begin1 == -1);
  assert(begin0 - mid1 == -2);

  // Sentinel minus.
  assert(std::default_sentinel - sv_zero.begin() == std::ranges::distance(sv_zero));
  assert(sv_zero.begin() - std::default_sentinel == -std::ranges::distance(sv_zero));
  assert(std::default_sentinel - sv_zero.end() == 0);
  assert(sv_zero.end() - std::default_sentinel == 0);

  return true;
}

template <std::forward_iterator Iter>
constexpr bool test_forward_minus(Iter begin, Iter end) {
  using Base = BasicTestView<Iter, Iter>;

  auto base_zero = Base(begin, end);
  auto base_one  = Base(begin + 1, end);
  auto sv_zero   = std::ranges::stride_view(base_zero, 3);
  auto sv_one    = std::ranges::stride_view(base_one, 3);

  auto begin0 = sv_zero.begin();
  auto mid0   = begin0;
  ++mid0; // stride 3 -> value 4
  auto far0 = mid0;
  ++far0; // stride 3 -> value 7

  auto begin1 = sv_one.begin(); // value 2
  auto mid1   = begin1;
  ++mid1; // value 5

  // Forward range uses exact division (no ceil).
  assert(mid0 - begin0 == 1);
  assert(far0 - begin0 == 2);
  assert(begin1 - begin0 == 0);
  assert(mid1 - begin0 == 1);

  assert(begin0 - mid0 == -1);
  assert(begin0 - far0 == -2);

  // Sentinel minus.
  assert(std::default_sentinel - sv_zero.begin() == std::ranges::distance(sv_zero));
  assert(sv_zero.begin() - std::default_sentinel == -std::ranges::distance(sv_zero));

  return true;
}

constexpr bool test() {
  {
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    test_forward_minus(arr, arr + 11);

    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    test_forward_minus(vec.begin(), vec.end());
  }
  {
    // operator-(iterator, difference_type) -- only for random access ranges.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    using Base = BasicTestView<int*, int*>;

    auto base = Base(arr, arr + 10);
    auto sv   = std::ranges::stride_view(base, 3);

    auto it = sv.begin();
    ++it;
    ++it;
    ++it; // at index 9

    auto it2 = it - 2; // back to index 3
    assert(*it2 == 4);
    auto it3 = it - 3; // back to index 0
    assert(*it3 == 1);
  }
  {
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    test_non_forward_minus(SizedInputIter(arr), SizedInputIter(arr + 1), SizedInputIter(arr + 10));
  }
  {
    // Test end - begin on the same view where size % stride != 0.
    // 10 elements, stride 3: strided elements at 0,3,6,9.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 10), 3);
    auto b     = sv.begin();
    auto e     = sv.end();
    assert(e - b == 4);
    assert(b - e == -4);
    assert(b - b == 0);
    assert(e - e == 0);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
