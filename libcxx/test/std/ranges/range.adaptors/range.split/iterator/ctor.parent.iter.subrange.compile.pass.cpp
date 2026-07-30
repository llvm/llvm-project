//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr iterator(split_view& parent, iterator_t<V> current, subrange<iterator_t<V>> next);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <concepts>
#include <ranges>

#include "../types.h"

struct TracedMoveIter : ForwardIterBase<TracedMoveIter> {
  bool moved = false;

  constexpr TracedMoveIter()                      = default;
  constexpr TracedMoveIter(const TracedMoveIter&) = default;
  constexpr TracedMoveIter(TracedMoveIter&&) : moved{true} {}
  constexpr TracedMoveIter& operator=(TracedMoveIter&&)      = default;
  constexpr TracedMoveIter& operator=(const TracedMoveIter&) = default;
};

struct TracedMoveView : std::ranges::view_base {
  constexpr TracedMoveIter begin() const { return {}; }
  constexpr TracedMoveIter end() const { return {}; }
};

using SplitView = std::ranges::split_view<TracedMoveView, TracedMoveView>;
using SplitIter = std::ranges::iterator_t<SplitView>;

static_assert(!std::constructible_from<SplitIter, SplitView, TracedMoveIter, std::ranges::subrange<TracedMoveIter>>);
