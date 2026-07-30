//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit iterator(W value);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <concepts>
#include <ranges>

#include "../types.h"

using IntIter = std::ranges::iterator_t<std::ranges::iota_view<int>>;

static_assert(!std::constructible_from<IntIter, int>);
static_assert(!std::convertible_to<int, IntIter>);

using SomeIntIter = std::ranges::iterator_t<std::ranges::iota_view<SomeInt>>;

static_assert(!std::constructible_from<SomeIntIter, SomeInt>);
static_assert(!std::convertible_to<SomeInt, SomeIntIter>);
