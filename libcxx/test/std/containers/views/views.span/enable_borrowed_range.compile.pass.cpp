//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template<class ElementType, size_t Extent>
// inline constexpr bool ranges::enable_borrowed_range<
//     span<ElementType, Extent>> = true;

#include <span>

#include "test_macros.h"

void test() {
  static_assert(std::ranges::enable_borrowed_range<std::span<int, 0> >);
  static_assert(std::ranges::enable_borrowed_range<std::span<int, 42> >);
  static_assert(std::ranges::enable_borrowed_range<std::span<int, std::dynamic_extent> >);
  static_assert(!std::ranges::enable_borrowed_range<std::span<int, 42>&>);
  static_assert(!std::ranges::enable_borrowed_range<std::span<int, 42> const>);
}
