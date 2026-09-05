//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<V>;
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const V>;

#include <cassert>
#include <ranges>

#include "test_iterators.h"

// forward_iterator + sentinel end so that ranges::size doesn't apply,
// making these ranges approximately_sized only via their member reserve_hint().

struct ConstReserveHintView : std::ranges::view_base {
  bool* hint_called;
  constexpr auto begin() const { return forward_iterator<int*>(nullptr); }
  constexpr auto end() const { return sentinel_wrapper<forward_iterator<int*>>(forward_iterator<int*>(nullptr)); }

  constexpr unsigned int reserve_hint() const {
    *hint_called = true;
    return 3;
  }
};

struct NonConstReserveHintView : std::ranges::view_base {
  bool* hint_called;
  constexpr auto begin() const { return forward_iterator<int*>(nullptr); }
  constexpr auto end() const { return sentinel_wrapper<forward_iterator<int*>>(forward_iterator<int*>(nullptr)); }

  constexpr unsigned int reserve_hint() {
    *hint_called = true;
    return 5;
  }
};

struct NoReserveHintView : std::ranges::view_base {
  constexpr auto begin() const { return forward_iterator<int*>(nullptr); }
  constexpr auto end() const { return sentinel_wrapper<forward_iterator<int*>>(forward_iterator<int*>(nullptr)); }
};

template <class T>
concept HasReserveHint = requires(T v) { v.reserve_hint(); };

static_assert(!std::ranges::sized_range<std::ranges::as_rvalue_view<ConstReserveHintView>>);
static_assert(!std::ranges::sized_range<std::ranges::as_rvalue_view<NonConstReserveHintView>>);
static_assert(!std::ranges::sized_range<std::ranges::as_rvalue_view<NoReserveHintView>>);

static_assert(HasReserveHint<std::ranges::as_rvalue_view<ConstReserveHintView>>);
static_assert(HasReserveHint<const std::ranges::as_rvalue_view<ConstReserveHintView>>);
static_assert(HasReserveHint<std::ranges::as_rvalue_view<NonConstReserveHintView>>);
static_assert(!HasReserveHint<const std::ranges::as_rvalue_view<NonConstReserveHintView>>);
static_assert(!HasReserveHint<std::ranges::as_rvalue_view<NoReserveHintView>>);
static_assert(!HasReserveHint<const std::ranges::as_rvalue_view<NoReserveHintView>>);

constexpr bool test() {
  {
    bool hint_called = false;
    std::ranges::as_rvalue_view view(ConstReserveHintView{{}, &hint_called});
    assert(view.reserve_hint() == 3);
    assert(hint_called);
  }

  {
    bool hint_called = false;
    std::ranges::as_rvalue_view view(NonConstReserveHintView{{}, &hint_called});
    assert(view.reserve_hint() == 5);
    assert(hint_called);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
