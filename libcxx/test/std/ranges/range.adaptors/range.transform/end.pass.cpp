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

// constexpr sentinel<false> end();
// constexpr iterator<false> end() requires common_range<V>;
// constexpr sentinel<true> end() const
//   requires range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;
// constexpr iterator<true> end() const
//   requires common_range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;

#include <ranges>

#include "test_macros.h"
#include "types.h"

template<class T>
concept EndInvocable = requires(T t) { t.end(); };

template<class T>
concept EndIsIter = requires(T t) { ++t.end(); };

constexpr bool test() {
  {
    std::ranges::transform_view transformView(MoveOnlyView{}, PlusOneMutable{});
    assert(transformView.end().base() == globalBuff + 8);
  }

  {
    std::ranges::transform_view transformView(ForwardView{}, PlusOneMutable{});
    assert(transformView.end().base().base() == globalBuff + 8);
  }

  {
    std::ranges::transform_view transformView(InputView{}, PlusOneMutable{});
    assert(transformView.end().base() == globalBuff + 8);
  }

  {
    const std::ranges::transform_view transformView(MoveOnlyView{}, PlusOne{});
    assert(transformView.end().base() == globalBuff + 8);
  }

  static_assert(!EndInvocable<const std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>);
  static_assert( EndInvocable<      std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>);
  static_assert( EndInvocable<const std::ranges::transform_view<MoveOnlyView, PlusOne>>);
  static_assert(!EndInvocable<const std::ranges::transform_view<InputView, PlusOneMutable>>);
  static_assert( EndInvocable<      std::ranges::transform_view<InputView, PlusOneMutable>>);
  static_assert( EndInvocable<const std::ranges::transform_view<InputView, PlusOne>>);

  static_assert(!EndIsIter<const std::ranges::transform_view<InputView, PlusOne>>);
  static_assert(!EndIsIter<      std::ranges::transform_view<InputView, PlusOneMutable>>);
  static_assert( EndIsIter<const std::ranges::transform_view<MoveOnlyView, PlusOne>>);
  static_assert( EndIsIter<      std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
