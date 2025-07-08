//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <ranges>

#include <type_traits>
#include <vector>

#include "../types.h"
#include "test_iterators.h"

constexpr bool test() {
  { // Regular conversion from `!Const` to `Const` sentinel
    using Inner            = BasicVectorView<int, ViewProperties{.common = false}, forward_iterator>;
    std::vector<Inner> vec = {Inner{11, 12}, Inner{13, 14}};

    std::ranges::join_with_view jwv(vec, 0);
    using JWV = decltype(jwv);
    static_assert(!std::ranges::common_range<JWV>);

    using Sent  = std::ranges::sentinel_t<JWV>;
    using CSent = std::ranges::sentinel_t<const JWV>;
    static_assert(!std::same_as<Sent, CSent>);

    Sent se                    = jwv.end();
    [[maybe_unused]] CSent cse = se;
  }

  { // Test conversion from `Const` to `!Const` (should be invalid)
    using Inner   = BasicVectorView<int, ViewProperties{.common = false}, forward_iterator>;
    using V       = std::vector<Inner>;
    using Pattern = std::ranges::single_view<int>;
    using JWV     = std::ranges::join_with_view<std::views::all_t<V>, Pattern>;
    static_assert(!std::ranges::common_range<JWV>);

    using Sent  = std::ranges::sentinel_t<JWV>;
    using CSent = std::ranges::sentinel_t<const JWV>;
    static_assert(!std::convertible_to<CSent, Sent>);
    static_assert(!std::constructible_from<Sent, CSent>);
  }

  { // When `convertible_to<sentinel_t<V>, sentinel_t<Base>>` is not modeled
    using V       = ConstOppositeView<std::vector<long>>;
    using Pattern = std::ranges::single_view<long>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;
    static_assert(!std::ranges::common_range<JWV>);

    using Sent  = std::ranges::sentinel_t<JWV>;
    using CSent = std::ranges::sentinel_t<const JWV>;
    static_assert(!std::convertible_to<CSent, Sent>);
    static_assert(!std::constructible_from<Sent, CSent>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
