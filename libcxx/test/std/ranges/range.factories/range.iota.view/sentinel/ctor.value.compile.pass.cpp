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

#include <ranges>
#include <type_traits>

#include "test_macros.h"

#include "../types.h"

using Sent = std::ranges::sentinel_t<std::ranges::iota_view<SomeInt, IntSentinelWith<SomeInt>>>;
static_assert(!std::is_constructible_v<Sent, IntSentinelWith<SomeInt>>);
static_assert(!std::is_convertible_v<IntSentinelWith<SomeInt>, Sent>);
