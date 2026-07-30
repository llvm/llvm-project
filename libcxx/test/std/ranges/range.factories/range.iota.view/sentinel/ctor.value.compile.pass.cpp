//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit sentinel(Bound bound);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <concepts>
#include <ranges>

#include "../types.h"

using Sent = std::ranges::sentinel_t<std::ranges::iota_view<SomeInt, IntSentinelWith<SomeInt>>>;

static_assert(!std::constructible_from<Sent, IntSentinelWith<SomeInt>>);
static_assert(!std::convertible_to<IntSentinelWith<SomeInt>, Sent>);
