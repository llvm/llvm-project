//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator+=(difference_type n)

#include <cassert>
#include <ranges>
#include <type_traits>
#include <vector>

#include "../types.h"
#include "test_iterators.h"

template <class T>
concept CanPlus =
    std::is_same_v<T&, decltype(std::declval<T>() += std::declval<typename T::difference_type>())> &&
    requires(T& t, typename T::difference_type u) { t += u; };

// Make sure that we cannot use += on a stride view iterator
// over an input view.(sized sentinel)
using InputView = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideViewOverInputViewIterator = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;

static_assert(std::ranges::input_range<InputView>);
static_assert(!CanPlus<StrideViewOverInputViewIterator>);

// Make sure that we cannot use += on a stride view iterator
// over a forward view.(sized sentinel)
using ForwardView                       = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideViewOverForwardViewIterator = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;

static_assert(std::ranges::forward_range<ForwardView>);
static_assert(!CanPlus<StrideViewOverForwardViewIterator>);

// Make sure that we cannot use += on a stride view iterator
// over a bidirectional view. (sized sentinel)
using BidirectionalView = BasicTestView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
using StrideViewOverBidirectionalViewIterator = std::ranges::iterator_t<std::ranges::stride_view<BidirectionalView>>;

static_assert(std::ranges::bidirectional_range<BidirectionalView>);
static_assert(!CanPlus<StrideViewOverBidirectionalViewIterator>);

// Make sure that we can use += on a stride view iterator
// over a random access view. (non sized sentinel)
template <typename RandomAccessIterator = random_access_iterator<int*>>
using RandomAccessView                       = BasicTestView<RandomAccessIterator>;
using StrideViewOverRandomAccessViewIterator = std::ranges::iterator_t<std::ranges::stride_view<RandomAccessView<>>>;

static_assert(std::ranges::random_access_range<RandomAccessView<>>);
static_assert(CanPlus<StrideViewOverRandomAccessViewIterator>);

constexpr bool test() {
  std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  using Iter = std::vector<int>::iterator;
  using Diff = Iter::difference_type;

  auto begin    = vec.begin();
  auto end      = vec.end();
  Diff distance = 4;

  using Base = RandomAccessView<Iter>;
  static_assert(std::ranges::random_access_range<Base>);

  // += with stride 1: advancing by distance matches starting at begin + distance.
  auto base       = Base(begin, end);
  auto sv         = std::ranges::stride_view(base, 1);

  auto base_offset = Base(begin + distance, end);
  auto sv_offset   = std::ranges::stride_view(base_offset, 1);

  auto it        = sv.begin();
  auto it_offset = sv_offset.begin();

  auto it_after = it += distance;
  assert(*it == *it_offset);
  assert(*it_after == *it_offset);

  // += past the end, then -= back: the remainder is handled correctly.
  auto big_step      = (end - 1) - begin;
  auto sv_big_step   = std::ranges::stride_view(base, big_step);
  it                 = sv_big_step.begin();

  // This += should move us into a position where the stride doesn't evenly divide the range.
  // Do a -= 1 here to confirm that the remainder is taken into account.
  it += 2;
  it -= 1;
  assert(*it == *(sv.begin() + big_step));
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
