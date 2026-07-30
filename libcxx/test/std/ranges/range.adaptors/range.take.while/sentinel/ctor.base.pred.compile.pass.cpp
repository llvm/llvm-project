//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit sentinel(sentinel_t<Base> end, const Pred* pred);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <concepts>
#include <ranges>

struct Sent {
  int i;

  friend constexpr bool operator==(int* iter, const Sent& s) { return s.i > *iter; }
};

struct Range : std::ranges::view_base {
  int* begin() const;
  Sent end();
};

struct Pred {
  bool operator()(int i) const;
};

using Sentinel = std::ranges::sentinel_t<std::ranges::take_while_view<Range, Pred>>;

static_assert(!std::constructible_from<Sentinel, std::ranges::sentinel_t<Range>, const Pred*>);
