//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr V base() const& requires copy_constructible<V>;
// constexpr V base() &&;

#include <ranges>

#include <cassert>
#include <utility>

#include "test_macros.h"
#include "types.h"

constexpr bool hasLValueQualifiedBase(auto&& view) {
  return requires { view.base(); };
}

constexpr bool test() {
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    CopyableView view{buf, buf + 8};
    std::ranges::common_view<CopyableView> common(view);
    assert(common.base().begin_ == buf);
    assert(std::move(common).base().begin_ == buf);

    ASSERT_SAME_TYPE(decltype(common.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(std::move(common).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(common));
  }

  {
    MoveOnlyView view{buf, buf + 8};
    std::ranges::common_view<MoveOnlyView> common(std::move(view));
    assert(std::move(common).base().begin_ == buf);

    ASSERT_SAME_TYPE(decltype(std::move(common).base()), MoveOnlyView);
    static_assert(!hasLValueQualifiedBase(common));
  }

  {
    CopyableView view{buf, buf + 8};
    const std::ranges::common_view<CopyableView> common(view);
    assert(common.base().begin_ == buf);
    assert(std::move(common).base().begin_ == buf);

    ASSERT_SAME_TYPE(decltype(common.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(std::move(common).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(common));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
