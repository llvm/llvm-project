//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<V>
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const V>

#include <cassert>
#include <ranges>

#include "types.h"

template <class T>
concept ReserveHintInvocable = requires(T t) { t.reserve_hint(); };

constexpr bool test() {
  {
    std::ranges::transform_view transformView(ApproximatelySizedView{5}, PlusOne{});
    assert(transformView.reserve_hint() == 5);
  }

  static_assert(ReserveHintInvocable<std::ranges::transform_view<ApproximatelySizedView, PlusOne>>);
  static_assert(ReserveHintInvocable<const std::ranges::transform_view<ApproximatelySizedView, PlusOne>>);

  static_assert(ReserveHintInvocable<std::ranges::transform_view<ApproximatelySizedNotConstView, PlusOne>>);
  static_assert(!ReserveHintInvocable<const std::ranges::transform_view<ApproximatelySizedNotConstView, PlusOne>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
