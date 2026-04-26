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
concept CanPlus = std::is_same_v<T&, decltype(std::declval<T>() += std::declval<typename T::difference_type>())> &&
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

  auto begin = vec.begin();
  auto end   = vec.end();

  using Base = RandomAccessView<Iter>;
  static_assert(std::ranges::random_access_range<Base>);
  auto base = Base(begin, end);

  // += with stride 1: advancing by distance matches starting at begin + distance.
  {
    auto sv      = std::ranges::stride_view(base, 1);
    auto it      = sv.begin();
    auto& result = (it += 4);
    assert(&result == &it);

    auto it2 = std::ranges::stride_view(Base(begin + 4, end), 1).begin();
    assert(*it == *it2);
  }

  // += past the end, then -= back: the remainder is handled correctly.
  {
    auto sv = std::ranges::stride_view(base, (end - begin) - 1);
    auto it = sv.begin();

    // This += should move us into a position where the stride doesn't evenly divide the range.
    // Do a -= 1 here to confirm that the remainder is taken into account.
    it += 2;
    it -= 1;
    assert(*it == *(sv.begin() + 1));
  }

  // += 0 is a no-op.
  {
    auto sv = std::ranges::stride_view(base, 3);
    auto it = sv.begin();
    it += 1; // at index 3
    assert(*it == *(begin + 3));
    it += 0;
    assert(*it == *(begin + 3));
  }

  // += with negative n.
  {
    auto sv = std::ranges::stride_view(base, 3);
    auto it = sv.begin();
    it += 3; // at index 9
    assert(*it == *(begin + 9));
    it += -2; // back to index 3
    assert(*it == *(begin + 3));
    it += -1; // back to index 0
    assert(*it == *begin);
  }

  // += negative from end.
  {
    int arr[]        = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    using CommonBase = BasicTestView<int*, int*>;
    auto sv          = std::ranges::stride_view(CommonBase(arr, arr + 11), 3);
    auto it          = sv.end();
    it += -1; // last strided element (index 9)
    assert(*it == 10);
    it += -3; // back to begin (index 0)
    assert(*it == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
