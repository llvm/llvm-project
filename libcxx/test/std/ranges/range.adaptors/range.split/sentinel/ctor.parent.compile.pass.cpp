//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit sentinel(split_view& parent);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <ranges>

#include "../types.h"

using Range     = std::ranges::subrange<int*, sentinel_wrapper<int*>>;
using SplitView = std::ranges::split_view<Range, std::ranges::single_view<int>>;
using SplitSent = std::ranges::sentinel_t<SplitView>;

static_assert(!std::is_constructible_v<SplitSent, SplitView&>);
static_assert(!std::is_convertible_v<SplitView&, SplitSent>);
