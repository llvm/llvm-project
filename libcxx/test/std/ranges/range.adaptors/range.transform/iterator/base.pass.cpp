//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// transform_view::<iterator>::base

#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  {
    using TransformView = std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    TransformView tv;
    auto it = tv.begin();
    using It = decltype(it);
    ASSERT_SAME_TYPE(decltype(static_cast<It&>(it).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<It&&>(it).base()), int*);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&>(it).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&&>(it).base()), int* const&);
    ASSERT_NOEXCEPT(it.base());
    assert(base(it.base()) == globalBuff);
    assert(base(std::move(it).base()) == globalBuff);
  }
  {
    using TransformView = std::ranges::transform_view<InputView, PlusOneMutable>;
    TransformView tv;
    auto it = tv.begin();
    using It = decltype(it);
    ASSERT_SAME_TYPE(decltype(static_cast<It&>(it).base()), const cpp20_input_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(static_cast<It&&>(it).base()), cpp20_input_iterator<int*>);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&>(it).base()), const cpp20_input_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&&>(it).base()), const cpp20_input_iterator<int*>&);
    ASSERT_NOEXCEPT(it.base());
    assert(base(it.base()) == globalBuff);
    assert(base(std::move(it).base()) == globalBuff);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
