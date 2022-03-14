//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <ranges>

#include "test_macros.h"
#include "types.h"

template<class T>
concept SizeInvocable = requires(T t) { t.size(); };

constexpr bool test() {
  {
    std::ranges::transform_view transformView(MoveOnlyView{}, PlusOne{});
    assert(transformView.size() == 8);
  }

  {
    const std::ranges::transform_view transformView(MoveOnlyView{globalBuff, 4}, PlusOne{});
    assert(transformView.size() == 4);
  }

  static_assert(!SizeInvocable<std::ranges::transform_view<ForwardView, PlusOne>>);

  static_assert(SizeInvocable<std::ranges::transform_view<SizedSentinelNotConstView, PlusOne>>);
  static_assert(!SizeInvocable<const std::ranges::transform_view<SizedSentinelNotConstView, PlusOne>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
