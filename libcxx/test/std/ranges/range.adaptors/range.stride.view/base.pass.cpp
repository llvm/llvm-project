//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr _View base() const& requires copy_constructible<_View>;
// constexpr _View base() &&;

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "types.h"

template <typename T>
constexpr bool hasLValueQualifiedBase(T&& t) {
  return requires { t.base(); };
}

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int N = 8;
  {
    using CopyableInputView = CopyableView<cpp17_input_iterator<int*>>;
    auto str(std::ranges::stride_view<CopyableInputView>(
        CopyableInputView(cpp17_input_iterator<int*>(buff), cpp17_input_iterator<int*>(buff + N)), 1));
    assert(*str.base().begin() == *buff);
    assert(*(std::move(str)).base().begin() == *buff);

    ASSERT_SAME_TYPE(decltype(str.base()), CopyableInputView);
    ASSERT_SAME_TYPE(decltype(std::move(str).base()), CopyableInputView);
    static_assert(hasLValueQualifiedBase(str));
  }

  {
    using MoveOnlyInputView = MoveOnlyView<cpp17_input_iterator<int*>>;
    auto str(std::ranges::stride_view<MoveOnlyInputView>(
        MoveOnlyInputView(cpp17_input_iterator<int*>(buff), cpp17_input_iterator<int*>(buff + N)), 1));
    assert(*(std::move(str)).base().begin() == *buff);

    ASSERT_SAME_TYPE(decltype(std::move(str).base()), MoveOnlyInputView);
    static_assert(!hasLValueQualifiedBase(str));
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
