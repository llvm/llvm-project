//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// template<class View>
//   constexpr bool enable_borrowed_range<enumerate_view<View>>;

#include <cassert>
#include <ranges>

struct NonBorrowedRange : std::ranges::view_base {
  int* begin();
  int* end();
};

struct BorrowedRange : std::ranges::view_base {
  int* begin();
  int* end();
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange> = true;

static_assert(!std::ranges::borrowed_range<std::ranges::enumerate_view<NonBorrowedRange>>);
static_assert(std::ranges::borrowed_range<std::ranges::enumerate_view<BorrowedRange>>);
