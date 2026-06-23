//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test the libc++ extension that std::ranges::adjacent_view and std::views::adjacent are marked as [[nodiscard]].
#include <ranges>
#include <utility>
#include <functional>

#include "test_iterators.h"

template<size_t N>
    requires (N > 0)
struct NonCommonSimpleView<N> : std::ranges::adjacent_view<NonCommonSimpleView, N> {
  int* begin();
  int* begin() const;
  sized_sentinel<int*> end();
  sized_sentinel<int*> end() const;
};

static_assert(!std::ranges::common_range<View>);
static_assert(
    std::same_as<std::ranges::iterator_t<NonCommonSimpleView>, std::ranges::iterator_t<const NonCommonSimpleView>>);
static_assert(
    std::same_as<std::ranges::sentinel_t<NonCommonSimpleView>, std::ranges::sentinel_t<const NonCommonSimpleView>>);

void test() {
  auto v = NonCommonSimpleView<2>{} | std::views::transform(std::identity{});
}
