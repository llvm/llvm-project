//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// cartesian_product_view does not opt in to enable_borrowed_range, even when its underlying
// ranges are themselves borrowed ranges. The view stores its bases by value, so its iterators
// would dangle once the view is destroyed.

#include <ranges>
#include <span>

using BorrowedSpan = std::span<int>;
static_assert(std::ranges::borrowed_range<BorrowedSpan>);

// cartesian_product_view itself is never borrowed, even when its bases are.
static_assert(!std::ranges::borrowed_range<std::ranges::cartesian_product_view<BorrowedSpan>>);
static_assert(!std::ranges::borrowed_range<std::ranges::cartesian_product_view<BorrowedSpan, BorrowedSpan>>);
static_assert(
    !std::ranges::borrowed_range<std::ranges::cartesian_product_view<BorrowedSpan, BorrowedSpan, BorrowedSpan>>);
