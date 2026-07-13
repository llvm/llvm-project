//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr decltype(auto) operator[](difference_type n) const

#include <cassert>
#include <ranges>

#include "../types.h"
#include "test_iterators.h"

template <class T>
concept CanSubscript = requires(T& t) { t[0]; };

// Input view: no subscript.
using InputView       = BasicTestView<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>;
using StrideInputIter = std::ranges::iterator_t<std::ranges::stride_view<InputView>>;
static_assert(!CanSubscript<StrideInputIter>);

// Forward view: no subscript.
using FwdView       = BasicTestView<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>;
using StrideFwdIter = std::ranges::iterator_t<std::ranges::stride_view<FwdView>>;
static_assert(!CanSubscript<StrideFwdIter>);

// Bidirectional view: no subscript.
using BidirView       = BasicTestView<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>;
using StrideBidirIter = std::ranges::iterator_t<std::ranges::stride_view<BidirView>>;
static_assert(!CanSubscript<StrideBidirIter>);

// Random access view: subscript available.
using RAView       = BasicTestView<random_access_iterator<int*>>;
using StrideRAIter = std::ranges::iterator_t<std::ranges::stride_view<RAView>>;
static_assert(CanSubscript<StrideRAIter>);

constexpr bool test() {
  {
    // Subscript with stride 1.
    int arr[]  = {10, 20, 30, 40, 50};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 5), 1);
    auto it    = sv.begin();

    assert(it[0] == 10);
    assert(it[1] == 20);
    assert(it[2] == 30);
    assert(it[3] == 40);
    assert(it[4] == 50);
  }
  {
    // Subscript with stride 2.
    int arr[]  = {10, 20, 30, 40, 50};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 5), 2);
    auto it    = sv.begin();

    assert(it[0] == 10);
    assert(it[1] == 30);
    assert(it[2] == 50);
  }
  {
    // Subscript with stride 3.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 10), 3);
    auto it    = sv.begin();

    assert(it[0] == 1);
    assert(it[1] == 4);
    assert(it[2] == 7);
    assert(it[3] == 10);
  }
  {
    // Subscript from a non-begin position.
    int arr[]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 10), 3);
    auto it    = sv.begin();
    ++it; // now at index 3 (value 4)

    assert(it[0] == 4);
    assert(it[1] == 7);
    assert(it[2] == 10);
    assert(it[-1] == 1);
  }
  {
    // Verify return type is a reference.
    int arr[]  = {1, 2, 3};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 3), 1);
    auto it    = sv.begin();
    static_assert(std::is_same_v<decltype(it[0]), int&>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
