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

// transform_view::<iterator>::base

#include <ranges>

#include "test_macros.h"
#include "../types.h"

template<class It>
concept HasBase = requires(It it) {
  static_cast<It>(it).base();
};

constexpr bool test() {
  {
    using TransformView = std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    TransformView tv;
    auto begin = tv.begin();
    ASSERT_SAME_TYPE(decltype(begin.base()), int*);
    assert(begin.base() == globalBuff);
    ASSERT_SAME_TYPE(decltype(std::move(begin).base()), int*);
    assert(std::move(begin).base() == globalBuff);
  }
  {
    using TransformView = std::ranges::transform_view<InputView, PlusOneMutable>;
    TransformView tv;
    auto begin = tv.begin();
    static_assert(!HasBase<decltype(begin)&>);
    static_assert(HasBase<decltype(begin)&&>);
    static_assert(!HasBase<const decltype(begin)&>);
    static_assert(!HasBase<const decltype(begin)&&>);
    std::same_as<cpp20_input_iterator<int *>> auto it = std::move(begin).base();
    assert(base(it) == globalBuff);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
