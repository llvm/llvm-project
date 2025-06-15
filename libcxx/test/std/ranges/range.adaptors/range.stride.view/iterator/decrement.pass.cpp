//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr __iterator& operator--()
// constexpr __iterator operator--(int)

#include <ranges>
#include <vector>

#include "../types.h"

template <class T>
concept CanPostDecrement = std::is_same_v<T, decltype(std::declval<T>()--)> && requires(T& t) { t--; };
template <class T>
concept CanPreDecrement = std::is_same_v<T, decltype(--(std::declval<T>()))> && requires(T& t) { --t; };

// What operators are valid for an iterator derived from a stride view
// over an input view.
using InputView = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideViewOverInputViewIterator = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;

static_assert(!std::ranges::bidirectional_range<InputView>);
static_assert(!CanPostDecrement<StrideViewOverInputViewIterator>);
static_assert(!CanPreDecrement<StrideViewOverInputViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a forward view.
using ForwardView                       = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideViewOverForwardViewIterator = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;

static_assert(!std::ranges::bidirectional_range<ForwardView>);
static_assert(!CanPostDecrement<StrideViewOverForwardViewIterator>);
static_assert(!CanPostDecrement<StrideViewOverForwardViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a bidirectional view.
using BidirectionalView = BasicTestView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
using StrideViewOverBidirectionalViewIterator = std::ranges::iterator_t<std::ranges::stride_view<BidirectionalView>>;

static_assert(std::ranges::bidirectional_range<BidirectionalView>);
static_assert(CanPostDecrement<StrideViewOverBidirectionalViewIterator>);
static_assert(CanPostDecrement<StrideViewOverBidirectionalViewIterator>);

// What operators are valid for an iterator derived from a stride view
// over a random access view.
using RandomAccessView                       = BasicTestView<random_access_iterator<int*>>;
using StrideViewOverRandomAccessViewIterator = std::ranges::iterator_t<std::ranges::stride_view<RandomAccessView>>;

static_assert(std::ranges::bidirectional_range<RandomAccessView>);
static_assert(CanPostDecrement<StrideViewOverRandomAccessViewIterator>);
static_assert(CanPostDecrement<StrideViewOverRandomAccessViewIterator>);

template <typename Iter, typename Difference>
  requires(std::bidirectional_iterator<Iter>)
constexpr bool test_operator_decrement(Iter begin, Iter end, Difference delta) {
  using Base = BasicTestView<Iter, Iter>;

  auto base_view_offset_zero = Base(begin, end);
  // Because of the requires on the Iter template type, we are sure
  // that the type of sv_incr_one is a bidirectional range.
  auto sv_incr_diff = std::ranges::stride_view(base_view_offset_zero, delta);
  auto sv_incr_end  = sv_incr_diff.end();

  // Recreate the "missing" calculation here -- to make sure that it matches.
  auto missing = delta - (std::ranges::distance(base_view_offset_zero) % delta) % delta;

  auto sought = end + (missing - delta);

  assert(*sought == *(--sv_incr_end));
  assert(*sought == *(sv_incr_end));

  sv_incr_end = sv_incr_diff.end();
  sv_incr_end--;
  assert(*(end + (missing - delta)) == *(sv_incr_end));

  return true;
}

int main(int, char**) {
  constexpr int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  test_operator_decrement(vec.begin(), vec.end(), 3);
  test_operator_decrement(arr, arr + 11, 3);

  return 0;
}
