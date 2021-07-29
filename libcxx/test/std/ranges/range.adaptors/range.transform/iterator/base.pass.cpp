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

// transform_view::<iterator>::base

#include <ranges>

#include "test_macros.h"
#include "../types.h"

template<class V, class F>
concept BaseInvocable = requires(std::ranges::iterator_t<std::ranges::transform_view<V, F>> iter) {
  iter.base();
};

constexpr bool test() {
  {
    std::ranges::transform_view<ContiguousView, Increment> transformView;
    auto iter = std::move(transformView).begin();
    ASSERT_SAME_TYPE(int*, decltype(iter.base()));
    assert(iter.base() == globalBuff);
    ASSERT_SAME_TYPE(int*, decltype(std::move(iter).base()));
    assert(std::move(iter).base() == globalBuff);
  }

  {
    std::ranges::transform_view<InputView, Increment> transformView;
    auto iter = transformView.begin();
    assert(std::move(iter).base() == globalBuff);
    ASSERT_SAME_TYPE(cpp20_input_iterator<int *>, decltype(std::move(iter).base()));
  }

  static_assert(!BaseInvocable<InputView, Increment>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
