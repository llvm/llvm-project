//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// transform_view::<iterator>::operator{+,-}

#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  std::ranges::transform_view<MoveOnlyView, PlusOneMutable> transformView1;
  auto iter1 = std::move(transformView1).begin();
  std::ranges::transform_view<MoveOnlyView, PlusOneMutable> transformView2;
  [[maybe_unused]] auto iter2 = std::move(transformView2).begin();
  iter1 += 4;
  assert((iter1 + 1).base() == globalBuff + 5);
  assert((1 + iter1).base() == globalBuff + 5);
  assert((iter1 - 1).base() == globalBuff + 3);
  LIBCPP_ASSERT(iter1 - iter2 == 4);
  assert((iter1 + 2) - 2 == iter1);
  assert((iter1 - 2) + 2 == iter1);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
