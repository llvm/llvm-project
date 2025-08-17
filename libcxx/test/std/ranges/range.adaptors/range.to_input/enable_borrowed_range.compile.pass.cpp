//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class to_input_view

//  template<class V>
//    constexpr bool enable_borrowed_range<to_input_view<V>> =
//      enable_borrowed_range<V>;

#include <ranges>
#include <tuple>

struct BorrowedView : std::ranges::view_base {
  int* begin();
  int* end();
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedView> = true;

static_assert(std::ranges::borrowed_range<std::ranges::to_input_view<BorrowedView>>);

struct NonBorrowedView : std::ranges::view_base {
  int* begin();
  int* end();
};

static_assert(!std::ranges::borrowed_range<std::ranges::to_input_view<NonBorrowedView>>);
