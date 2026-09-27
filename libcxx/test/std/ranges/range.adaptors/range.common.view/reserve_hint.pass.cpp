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
#include "test_range.h"
#include "types.h"

static_assert(HasReserveHint<std::ranges::common_view<ApproximatelySizedView>&>);
static_assert(HasReserveHint<std::ranges::common_view<ApproximatelySizedView> const&>);
static_assert(!HasReserveHint<std::ranges::common_view<CopyableView>&>);
static_assert(!HasReserveHint<std::ranges::common_view<CopyableView> const&>);
static_assert(HasReserveHint<std::ranges::common_view<NonConstApproximatelySizedView>&>);
static_assert(!HasReserveHint<std::ranges::common_view<NonConstApproximatelySizedView> const&>);

constexpr bool test() {
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    ApproximatelySizedView view(buf, buf + 8, 5);
    std::ranges::common_view<ApproximatelySizedView> common(view);
    assert(common.reserve_hint() == 5);
  }

  {
    ApproximatelySizedView view(buf, buf + 8, 5);
    const std::ranges::common_view<ApproximatelySizedView> common(view);
    assert(common.reserve_hint() == 5);
  }

  {
    NonConstApproximatelySizedView view(buf, buf + 8, 5);
    std::ranges::common_view<NonConstApproximatelySizedView> common(view);
    assert(common.reserve_hint() == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
