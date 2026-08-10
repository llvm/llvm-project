//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr V base() const& requires copy_constructible<V>;
// constexpr V base() &&;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "types.h"

constexpr bool hasLValueQualifiedBase(auto&& view) {
    return requires { view.base(); };
}

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 0);
    assert(tv.base().ptr_ == buffer);
    assert(std::move(tv).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(tv.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(std::move(tv).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(tv));
  }

  {
    std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 1);
    assert(std::move(tv).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(std::move(tv).base()), MoveOnlyView);
    static_assert(!hasLValueQualifiedBase(tv));
  }

  {
    const std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 2);
    assert(tv.base().ptr_ == buffer);
    assert(std::move(tv).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(tv.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(std::move(tv).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(tv));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
