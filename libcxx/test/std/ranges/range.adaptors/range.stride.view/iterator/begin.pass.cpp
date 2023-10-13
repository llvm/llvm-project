//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view::iterator

#include "../test.h"
#include "__iterator/concepts.h"
#include <ranges>
#include <type_traits>

constexpr bool iterator_concept_test() {
  {
    int arr[] = {1, 2, 3};
    // Iterator of stride over random access view should have random access concept.
    auto rav = RandomAccessView(arr, arr + 3);
    auto str = std::ranges::stride_view<RandomAccessView>(rav, 1);
    static_assert(std::random_access_iterator<decltype(rav.begin())>);
    static_assert(std::random_access_iterator<decltype(str.begin())>);
  }

  {
    int arr[] = {1, 2, 3};
    // Iterator of stride over bidirectional view should have bidirectional view concept.
    auto rav = BidirView(arr, arr + 3);
    auto str = std::ranges::stride_view<BidirView>(rav, 1);
    static_assert(std::bidirectional_iterator<decltype(rav.begin())>);
    static_assert(std::bidirectional_iterator<decltype(str.begin())>);
    static_assert(!std::random_access_iterator<decltype(rav.begin())>);
    static_assert(!std::random_access_iterator<decltype(str.begin())>);
  }

  {
    int arr[] = {1, 2, 3};
    // Iterator of stride over forward view should have forward view concept.
    auto rav = ForwardView(arr, arr + 3);
    auto str = std::ranges::stride_view<ForwardView>(rav, 1);
    static_assert(std::forward_iterator<decltype(rav.begin())>);
    static_assert(std::forward_iterator<decltype(str.begin())>);
    static_assert(!std::bidirectional_iterator<decltype(rav.begin())>);
    static_assert(!std::bidirectional_iterator<decltype(str.begin())>);
    static_assert(!std::random_access_iterator<decltype(rav.begin())>);
    static_assert(!std::random_access_iterator<decltype(str.begin())>);
  }

  {
    int arr[] = {1, 2, 3};
    // Iterator of stride over input view should have input view concept.
    auto rav = InputView(arr, arr + 3);
    auto str = std::ranges::stride_view<InputView>(rav, 1);
    static_assert(std::input_iterator<decltype(rav.begin())>);
    static_assert(std::input_iterator<decltype(str.begin())>);
    static_assert(!std::forward_iterator<decltype(rav.begin())>);
    static_assert(!std::forward_iterator<decltype(str.begin())>);
    static_assert(!std::bidirectional_iterator<decltype(rav.begin())>);
    static_assert(!std::bidirectional_iterator<decltype(str.begin())>);
    static_assert(!std::random_access_iterator<decltype(rav.begin())>);
    static_assert(!std::random_access_iterator<decltype(str.begin())>);
  }
  return true;
}

int main(int, char**) {
  iterator_concept_test();
  static_assert(iterator_concept_test());
  return 0;
}
