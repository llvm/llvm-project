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

// transform_view::<iterator>::operator{++,--,+=,-=}

#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  std::ranges::transform_view<ContiguousView, Increment> transformView;
  auto iter = std::move(transformView).begin();
  assert((++iter).base() == globalBuff + 1);

  assert((iter++).base() == globalBuff + 1);
  assert(iter.base() == globalBuff + 2);

  assert((--iter).base() == globalBuff + 1);
  assert((iter--).base() == globalBuff + 1);
  assert(iter.base() == globalBuff);

  // Check that decltype(InputIter++) == void.
  ASSERT_SAME_TYPE(decltype(
    std::declval<std::ranges::iterator_t<std::ranges::transform_view<InputView, Increment>>>()++),
    void);

  assert((iter += 4).base() == globalBuff + 4);
  assert((iter -= 3).base() == globalBuff + 1);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
