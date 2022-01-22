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

// transform_view::<iterator>::operator[]

#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  std::ranges::transform_view transformView1(MoveOnlyView{buff}, PlusOneMutable{});
  auto iter1 = std::move(transformView1).begin() + 1;
  assert(iter1[0] == 2);
  assert(iter1[4] == 6);

  ASSERT_NOT_NOEXCEPT(
    std::declval<std::ranges::iterator_t<std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>>()[0]);
  LIBCPP_ASSERT_NOEXCEPT(
    std::declval<std::ranges::iterator_t<std::ranges::transform_view<MoveOnlyView, PlusOneNoexcept>>>()[0]);

  ASSERT_SAME_TYPE(
    int,
    decltype(std::declval<std::ranges::transform_view<RandomAccessView, PlusOneMutable>>().begin()[0]));
  ASSERT_SAME_TYPE(
    int&,
    decltype(std::declval<std::ranges::transform_view<RandomAccessView, Increment>>().begin()[0]));
  ASSERT_SAME_TYPE(
    int&&,
    decltype(std::declval<std::ranges::transform_view<RandomAccessView, IncrementRvalueRef>>().begin()[0]));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
