//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr outer-iterator(Parent& parent, iterator_t<Base> current);
//   requires forward_range<Base>

// The constructor is now `private` (exposition-only) per P3059R2.

#include <ranges>
#include <type_traits>

#include "../types.h"

static_assert(std::ranges::forward_range<SplitViewForward>);

static_assert(!std::is_constructible_v<OuterIterForward, SplitViewForward&, std::ranges::iterator_t<ForwardView>>);
