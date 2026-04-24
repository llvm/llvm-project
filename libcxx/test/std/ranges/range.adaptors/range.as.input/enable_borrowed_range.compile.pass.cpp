//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//  template<input_range V>
//    requires view<V>
//  class as_input_view : public view_interface<as_input_view<V>>

//  template<class V>
//    constexpr bool enable_borrowed_range<as_input_view<V>> =
//      enable_borrowed_range<V>;

#include <ranges>
#include <tuple>

struct BorrowedView : std::ranges::view_base {
  int* begin();
  int* end();
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedView> = true;

static_assert(std::ranges::borrowed_range<std::ranges::as_input_view<BorrowedView>>);

struct NonBorrowedView : std::ranges::view_base {
  int* begin();
  int* end();
};

static_assert(!std::ranges::borrowed_range<std::ranges::as_input_view<NonBorrowedView>>);
