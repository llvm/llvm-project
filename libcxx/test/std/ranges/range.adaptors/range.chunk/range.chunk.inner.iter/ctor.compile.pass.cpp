//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range:
//     inner_iterator(inner_iterator&&) = default;
//     inner_iterator& operator=(inner_iterator&&) = default;

#include <concepts>
#include <ranges>

#include "../types.h"

using InnerIterator = std::ranges::iterator_t<std::ranges::range_reference_t<std::ranges::chunk_view<input_span<int>>>>;

static_assert(!std::default_initializable<InnerIterator>);
static_assert(!std::copy_constructible<InnerIterator>);
static_assert(!std::assignable_from<InnerIterator&, const InnerIterator&>);
static_assert(std::move_constructible<InnerIterator>);
static_assert(std::assignable_from<InnerIterator&, InnerIterator>);
