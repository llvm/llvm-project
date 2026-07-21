//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit sentinel(sentinel_t<Base> end);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <ranges>
#include <type_traits>

struct Sent {
  int i;

  friend constexpr bool operator==(std::tuple<int>*, const Sent&) { return true; }
};

struct Range : std::ranges::view_base {
  std::tuple<int>* begin() const;
  Sent end();
};

static_assert(!std::is_constructible_v<std::ranges::sentinel_t<std::ranges::elements_view<Range, 0>>, Sent>);
static_assert(!std::is_convertible_v<Sent, std::ranges::sentinel_t<std::ranges::elements_view<Range, 0>>>);
