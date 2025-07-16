//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class cache_latest_view

// Test that std::ranges::cache_latest_view is not std::ranges::borrowed_range.

#include <ranges>

struct BorrowedView : std::ranges::view_base {
  int* begin();
  int* end();
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedView> = true;

static_assert(!std::ranges::borrowed_range<std::ranges::cache_latest_view<BorrowedView>>);

struct NonBorrowedView : std::ranges::view_base {
  int* begin();
  int* end();
};

static_assert(!std::ranges::borrowed_range<std::ranges::cache_latest_view<NonBorrowedView>>);
