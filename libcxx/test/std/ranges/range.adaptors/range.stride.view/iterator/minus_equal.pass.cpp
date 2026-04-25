//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator-=(difference_type n)

#include <cassert>
#include <ranges>
#include <type_traits>

#include "../types.h"
#include "test_iterators.h"

template <class T>
concept CanMinusEqual = std::is_same_v<T&, decltype(std::declval<T>() -= 1)> && requires(T& t) { t -= 1; };

// Input view: no -=.
using InputView = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideViewOverInputViewIter = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;
static_assert(!CanMinusEqual<StrideViewOverInputViewIter>);

// Forward view: no -=.
using ForwardView                   = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideViewOverForwardViewIter = std::ranges::iterator_t<std::ranges::stride_view<ForwardView>>;
static_assert(!CanMinusEqual<StrideViewOverForwardViewIter>);

// Bidirectional view: no -=.
using BidirView = BasicTestView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
using StrideViewOverBidirViewIter = std::ranges::iterator_t<std::ranges::stride_view<BidirView>>;
static_assert(!CanMinusEqual<StrideViewOverBidirViewIter>);

// Random access view: has -=.
using RAView                   = BasicTestView<random_access_iterator<int*>>;
using StrideViewOverRAViewIter = std::ranges::iterator_t<std::ranges::stride_view<RAView>>;
static_assert(CanMinusEqual<StrideViewOverRAViewIter>);

constexpr bool test() {
  {
    // Basic -= test with stride 1.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 10), 1);
    auto it    = sv.begin();
    it += 5;
    assert(*it == 6);
    it -= 3;
    assert(*it == 3);
    it -= 2;
    assert(*it == 1);
  }
  {
    // -= test with stride 3.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 10), 3);
    auto it    = sv.begin();
    it += 3; // at index 9 (value 10)
    assert(*it == 10);
    it -= 1; // at index 6 (value 7)
    assert(*it == 7);
    it -= 2; // at index 0 (value 1)
    assert(*it == 1);
  }
  {
    // -= when the stride doesn't evenly divide the range: stride past the end, then back.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view<Base>(Base(arr, arr + 10), 7);
    auto it    = sv.end();
    it -= 1; // back to last strided element (value 8)
    assert(*it == 8);
    it -= 1; // back to first element (value 1)
    assert(*it == 1);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
