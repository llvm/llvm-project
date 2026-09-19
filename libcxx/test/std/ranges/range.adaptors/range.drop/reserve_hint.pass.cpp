//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//   requires approximately_sized_range<V>
// constexpr auto reserve_hint() const
//   requires approximately_sized_range<const V>

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "types.h"

template <class T>
concept ReserveHintInvocable = requires(std::ranges::drop_view<T> t) { t.reserve_hint(); };

constexpr bool test() {
  // approximately_sized_range<V>
  std::ranges::drop_view dropView1(MoveOnlyView(), 4);
  assert(dropView1.reserve_hint() == 4);

  // approximately_sized_range<V>
  std::ranges::drop_view dropView2(MoveOnlyView(), 0);
  assert(dropView2.reserve_hint() == 8);

  // approximately_sized_range<const V>
  const std::ranges::drop_view dropView3(MoveOnlyView(), 8);
  assert(dropView3.reserve_hint() == 0);

  // approximately_sized_range<const V>
  const std::ranges::drop_view dropView4(MoveOnlyView(), 10);
  assert(dropView4.reserve_hint() == 0);

  // mutable-only approximately_sized_range
  std::ranges::drop_view dropView5(ApproximatelySizedNotConstView(8), 3);
  assert(dropView5.reserve_hint() == 5);
  static_assert(ReserveHintInvocable<ApproximatelySizedNotConstView>);
  static_assert(!ReserveHintInvocable<const ApproximatelySizedNotConstView>);

  // Because ForwardView is not approximately_sized_range.
  static_assert(!ReserveHintInvocable<ForwardView>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
