//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// transform_view::<iterator>::operator*

#include <ranges>

#include "test_macros.h"
#include "../types.h"

int main(int, char**) {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    std::ranges::transform_view transformView(ContiguousView{buff}, Increment{});
    assert(*transformView.begin() == 1);
  }

  static_assert(!noexcept(
    *std::declval<std::ranges::iterator_t<std::ranges::transform_view<ContiguousView, Increment>>>()));
  static_assert( noexcept(
    *std::declval<std::ranges::iterator_t<std::ranges::transform_view<ContiguousView, IncrementNoexcept>>>()));

  ASSERT_SAME_TYPE(
    int,
    decltype(*std::declval<std::ranges::transform_view<RandomAccessView, Increment>>().begin()));
  ASSERT_SAME_TYPE(
    int&,
    decltype(*std::declval<std::ranges::transform_view<RandomAccessView, IncrementRef>>().begin()));
  ASSERT_SAME_TYPE(
    int&&,
    decltype(*std::declval<std::ranges::transform_view<RandomAccessView, IncrementRvalueRef>>().begin()));

  return 0;
}
