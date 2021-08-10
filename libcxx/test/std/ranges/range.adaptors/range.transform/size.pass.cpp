//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
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
    std::ranges::transform_view transformView(ContiguousView{}, Increment{});
    assert(transformView.size() == 8);
  }

  {
    const std::ranges::transform_view transformView(ContiguousView{globalBuff, 4}, Increment{});
    assert(transformView.size() == 4);
  }

  static_assert(!SizeInvocable<std::ranges::transform_view<ForwardView, Increment>>);

  static_assert(SizeInvocable<std::ranges::transform_view<SizedSentinelNotConstView, Increment>>);
  static_assert(!SizeInvocable<const std::ranges::transform_view<SizedSentinelNotConstView, Increment>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
