//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// V models only input_range:
//   outer_iterator(outer_iterator&&) = default;
//   outer_iterator& operator=(outer_iterator&&) = default;

#include <concepts>
#include <ranges>

#include "../types.h"

using OuterIterator = std::ranges::iterator_t<std::ranges::chunk_view<input_span<int>>>;

static_assert(!std::default_initializable<OuterIterator>);
static_assert(!std::copy_constructible<OuterIterator>);
static_assert(!std::assignable_from<OuterIterator&, const OuterIterator&>);
static_assert(std::move_constructible<OuterIterator>);
static_assert(std::assignable_from<OuterIterator&, OuterIterator>);
