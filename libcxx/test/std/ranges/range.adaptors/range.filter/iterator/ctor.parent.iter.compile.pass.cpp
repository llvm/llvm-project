//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr std::ranges::filter_view::<iterator>(filter_view&, iterator_t<V>);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "../types.h"

using View           = minimal_view<int*, sentinel_wrapper<int*>>;
using ViewIter       = std::ranges::iterator_t<View>;
using FilterView     = std::ranges::filter_view<View, AlwaysTrue>;
using FilterIterator = std::ranges::iterator_t<FilterView>;

static_assert(!std::is_constructible_v<FilterIterator, FilterView&, ViewIter>);
