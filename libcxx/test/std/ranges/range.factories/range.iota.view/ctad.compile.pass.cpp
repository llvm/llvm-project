//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class W, class Bound>
//     requires (!is-integer-like<W> || !is-integer-like<Bound> ||
//               (is-signed-integer-like<W> == is-signed-integer-like<Bound>))
//     iota_view(W, Bound) -> iota_view<W, Bound>;

#include <ranges>
#include <cassert>
#include <concepts>

#include "test_macros.h"
#include "types.h"

template<class T, class U>
concept CanDeduce = requires(const T& t, const U& u) {
  std::ranges::iota_view(t, u);
};

void test() {
  static_assert(std::same_as<
    decltype(std::ranges::iota_view(0, 0)),
    std::ranges::iota_view<int, int>
  >);

  static_assert(std::same_as<
    decltype(std::ranges::iota_view(0)),
    std::ranges::iota_view<int, std::unreachable_sentinel_t>
  >);

  static_assert(std::same_as<
    decltype(std::ranges::iota_view(0, std::unreachable_sentinel)),
    std::ranges::iota_view<int, std::unreachable_sentinel_t>
  >);

  static_assert(std::same_as<
    decltype(std::ranges::iota_view(0, IntComparableWith(0))),
    std::ranges::iota_view<int, IntComparableWith<int>>
  >);

  static_assert( CanDeduce<int, int>);
  static_assert(!CanDeduce<int, unsigned>);
  static_assert(!CanDeduce<unsigned, int>);
}
