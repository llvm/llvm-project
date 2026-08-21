//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit sentinel(filter_view&);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "../types.h"

using View           = minimal_view<int*, sentinel_wrapper<int*>>;
using FilterView     = std::ranges::filter_view<View, AlwaysTrue>;
using FilterSentinel = std::ranges::sentinel_t<FilterView>;

static_assert(!std::is_constructible_v<FilterSentinel, FilterView>);
static_assert(!std::is_convertible_v<FilterView, FilterSentinel>);
