//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<R>
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const R>

#include <array>
#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

#include "test_iterators.h"

int globalBuff[8];

template <class T>
concept HasReserveHint = requires(T t) { t.reserve_hint(); };

constexpr bool test() {
  {
    struct NoReserveHint {
      bidirectional_iterator<int*> begin();
      bidirectional_iterator<int*> end();
    };
    using OwningView = std::ranges::owning_view<NoReserveHint>;
    static_assert(!HasReserveHint<OwningView&>);
    static_assert(!HasReserveHint<OwningView&&>);
    static_assert(!HasReserveHint<const OwningView&>);
    static_assert(!HasReserveHint<const OwningView&&>);
  }
  {
    struct ReserveHintMember {
      bidirectional_iterator<int*> begin();
      bidirectional_iterator<int*> end();
      int reserve_hint() const;
    };
    using OwningView = std::ranges::owning_view<ReserveHintMember>;
    static_assert(!std::ranges::sized_range<OwningView&>);
    static_assert(std::ranges::approximately_sized_range<OwningView&>);
    static_assert(!std::ranges::range<const OwningView&>); // no begin/end
    static_assert(HasReserveHint<OwningView&>);
    static_assert(HasReserveHint<OwningView&&>);
    static_assert(!HasReserveHint<const OwningView&>); // not a range, therefore no reserve_hint()
    static_assert(!HasReserveHint<const OwningView&&>);
  }
  {
    // Test an empty view.
    int a[] = {1};
    auto ov = std::ranges::owning_view(std::ranges::subrange(a, a));
    assert(ov.reserve_hint() == 0);
    assert(std::as_const(ov).reserve_hint() == 0);
  }
  {
    // Test a non-empty view.
    int a[] = {1};
    auto ov = std::ranges::owning_view(std::ranges::subrange(a, a + 1));
    assert(ov.reserve_hint() == 1);
    assert(std::as_const(ov).reserve_hint() == 1);
  }
  {
    // Test a non-view.
    std::array<int, 2> a = {1, 2};
    auto ov              = std::ranges::owning_view(std::move(a));
    assert(ov.reserve_hint() == 2);
    assert(std::as_const(ov).reserve_hint() == 2);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
