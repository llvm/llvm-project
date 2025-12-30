//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr _View base() const& requires copy_constructible<_View>;
// constexpr _View base() &&;

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "types.h"

template <typename T>
constexpr bool hasLValueQualifiedBase(T&& t) {
  // Thanks to forwarding references, t's type is
  // preserved from the caller. No matter the type of
  // the argument, when it is used here, t is an l value
  // (after all, it has a name). Therefore, this code
  // will test whether the l value const-ref-qualified
  // version of base is callable.
  return requires { t.base(); };
}

using CopyableInputView = CopyableView<cpp17_input_iterator<int*>>;
using MoveOnlyInputView = MoveOnlyView<cpp17_input_iterator<int*>>;

constexpr bool test() {
  int buff[]      = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int N = 8;

  // l-value ref qualified
  // const &
  {
    const auto str(std::ranges::stride_view<CopyableInputView>(
        CopyableInputView(cpp17_input_iterator<int*>(buff), cpp17_input_iterator<int*>(buff + N)), 1));

    static_assert(hasLValueQualifiedBase(str));

    std::same_as<CopyableInputView> decltype(auto) s = str.base();
    assert(*s.begin() == *buff);
  }

  // &
  {
    auto str(std::ranges::stride_view<CopyableInputView>(
        CopyableInputView(cpp17_input_iterator<int*>(buff), cpp17_input_iterator<int*>(buff + N)), 1));

    std::same_as<CopyableInputView> decltype(auto) s = str.base();
    assert(*s.begin() == *buff);

    static_assert(hasLValueQualifiedBase(str));
  }

  // r-value ref qualified
  // &&
  {
    auto str(std::ranges::stride_view<CopyableInputView>(
        CopyableInputView(cpp17_input_iterator<int*>(buff), cpp17_input_iterator<int*>(buff + N)), 1));

    static_assert(hasLValueQualifiedBase(str));

    std::same_as<CopyableInputView> decltype(auto) s = std::move(str).base();
    assert(*s.begin() == *buff);
  }

  // const &&
  {
    const auto str_a(std::ranges::stride_view<CopyableInputView>(
        CopyableInputView(cpp17_input_iterator<int*>(buff), cpp17_input_iterator<int*>(buff + N)), 1));

    static_assert(hasLValueQualifiedBase(str_a));

    std::same_as<CopyableInputView> decltype(auto) s = std::move(str_a).base();
    assert(*s.begin() == *buff);
  }

  // &&
  {
    auto str(std::ranges::stride_view<MoveOnlyInputView>(
        MoveOnlyInputView(cpp17_input_iterator<int*>(buff), cpp17_input_iterator<int*>(buff + N)), 1));

    // Because the base of the stride view is move only,
    // the const & version is not applicable and, therefore,
    // there is no l-value qualified base method.
    static_assert(!hasLValueQualifiedBase(str));

    std::same_as<MoveOnlyInputView> decltype(auto) s = std::move(str).base();
    assert(*s.begin() == *buff);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
