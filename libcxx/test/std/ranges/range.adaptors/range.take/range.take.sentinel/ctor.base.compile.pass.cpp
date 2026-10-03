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

#include "test_iterators.h"
#include "../types.h"

using TakeView = std::ranges::take_view<MoveOnlyView>;
using Sentinel = std::ranges::sentinel_t<TakeView>;

static_assert(!std::is_constructible_v<Sentinel, sentinel_wrapper<int*>>);
static_assert(!std::is_convertible_v<sentinel_wrapper<int*>, Sentinel>);
