//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <string_view>

// template<class charT, class traits>
// inline constexpr bool ranges::enable_borrowed_range<
//     basic_string_view<charT, traits>> = true;

#include <string_view>

#include "test_macros.h"

void test() {
  using std::ranges::enable_borrowed_range;
  static_assert(enable_borrowed_range<std::basic_string_view<char> >);
  static_assert(enable_borrowed_range<std::basic_string_view<wchar_t> >);
  static_assert(enable_borrowed_range<std::basic_string_view<char8_t> >);
}
