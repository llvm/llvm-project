//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// [range.cartesian.view] specifies no enable_borrowed_range for cartesian_product_view: its iterator holds a
// pointer to the parent view ([range.cartesian.iterator]), so iterators cannot outlive it.

#include <ranges>
#include <span>

using Borrowed = std::span<int>;
static_assert(std::ranges::borrowed_range<Borrowed>);

struct NonBorrowed : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};
static_assert(!std::ranges::borrowed_range<NonBorrowed>);

static_assert(!std::ranges::enable_borrowed_range<std::ranges::cartesian_product_view<Borrowed>>);
static_assert(!std::ranges::enable_borrowed_range<std::ranges::cartesian_product_view<Borrowed, Borrowed>>);
static_assert(!std::ranges::enable_borrowed_range<std::ranges::cartesian_product_view<Borrowed, Borrowed, Borrowed>>);
static_assert(!std::ranges::enable_borrowed_range<std::ranges::cartesian_product_view<Borrowed, NonBorrowed>>);

static_assert(std::ranges::range<std::ranges::cartesian_product_view<Borrowed>>);
static_assert(!std::ranges::borrowed_range<std::ranges::cartesian_product_view<Borrowed>>);
