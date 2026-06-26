//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   template <class V>
//   inline constexpr bool enable_borrowed_range<chunk_view<V>> =
//       forward_range<V> && enable_borrowed_range<V>;

#include <ranges>

#include "test_range.h"
#include "types.h"

// When V models only `input_range`.
static_assert(!std::ranges::enable_borrowed_range<std::ranges::chunk_view<input_span<int>>>);

// When V models at least `forward_range`.
static_assert(std::ranges::enable_borrowed_range<std::ranges::chunk_view<BorrowedView>>);
static_assert(!std::ranges::enable_borrowed_range<std::ranges::chunk_view<NonBorrowedView>>);
