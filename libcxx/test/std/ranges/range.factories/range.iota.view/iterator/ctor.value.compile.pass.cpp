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

#include <ranges>
#include <type_traits>

#include "../types.h"

using IntIter = std::ranges::iterator_t<std::ranges::iota_view<int>>;

static_assert(!std::is_constructible_v<IntIter, int>);
static_assert(!std::is_convertible_v<int, IntIter>);

using SomeIntIter = std::ranges::iterator_t<std::ranges::iota_view<SomeInt>>;

static_assert(!std::is_constructible_v<SomeIntIter, SomeInt>);
static_assert(!std::is_convertible_v<SomeInt, SomeIntIter>);
