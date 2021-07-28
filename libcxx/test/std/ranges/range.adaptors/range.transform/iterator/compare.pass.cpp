//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// transform_view::<iterator>::operator{<,>,<=,>=}

#include <ranges>
#include <compare>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  {
    std::ranges::transform_view<ContiguousView, Increment> transformView1;
    auto iter1 = std::move(transformView1).begin();
    std::ranges::transform_view<ContiguousView, Increment> transformView2;
    auto iter2 = std::move(transformView2).begin();
    assert(iter1 == iter2);
    assert(iter1 + 1 != iter2);
    assert(iter1 + 1 == iter2 + 1);

    assert(iter1 < iter1 + 1);
    assert(iter1 + 1 > iter1);
    assert(iter1 <= iter1 + 1);
    assert(iter1 <= iter2);
    assert(iter1 + 1 >= iter2);
    assert(iter1     >= iter2);
  }

// TODO: when three_way_comparable is implemented and std::is_eq is implemented,
// uncomment this.
//   {
//     std::ranges::transform_view<ThreeWayCompView, Increment> transformView1;
//     auto iter1 = transformView1.begin();
//     std::ranges::transform_view<ThreeWayCompView, Increment> transformView2;
//     auto iter2 = transformView2.begin();
//
//     assert(std::is_eq(iter1   <=> iter2));
//     assert(std::is_lteq(iter1 <=> iter2));
//     ++iter2;
//     assert(std::is_neq(iter1  <=> iter2));
//     assert(std::is_lt(iter1   <=> iter2));
//     assert(std::is_gt(iter2   <=> iter1));
//     assert(std::is_gteq(iter2 <=> iter1));
//
//     static_assert( std::three_way_comparable<std::iterator_t<std::ranges::transform_view<ThreeWayCompView, Increment>>>);
//     static_assert(!std::three_way_comparable<std::iterator_t<std::ranges::transform_view<ContiguousView, Increment>>>);
//   }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
