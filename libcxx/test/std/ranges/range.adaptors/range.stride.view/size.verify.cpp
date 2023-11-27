//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto size()

#include <cassert>
#include <ranges>

#include "test.h"

// There is no size member function on a stride view over a view that
// is *not* a sized range
static_assert(!std::ranges::sized_range<UnsizedBasicRange>);                           // expected-no-diagnostics
static_assert(!std::ranges::sized_range<std::ranges::stride_view<UnsizedBasicRange>>); // expected-no-diagnosticss

constexpr bool test() {
  {
    // Test with stride as exact multiple of number of elements in view strided over.
    constexpr auto iota_twelve = std::views::iota(0, 12);
    static_assert(std::ranges::sized_range<decltype(iota_twelve)>); // expected-no-diagnostics
    constexpr auto stride_iota_twelve = std::views::stride(iota_twelve, 3);
    static_assert(std::ranges::sized_range<decltype(stride_iota_twelve)>); // expected-no-diagnostics
    static_assert(4 == stride_iota_twelve.size(),
                  "Striding by 3 through a 12 member list has size 4."); // expected-no-diagnostics
  }

  {
    // Test with stride as inexact multiple of number of elements in view strided over.
    constexpr auto iota_twenty_two = std::views::iota(0, 22);
    static_assert(std::ranges::sized_range<decltype(iota_twenty_two)>); // expected-no-diagnostics
    constexpr auto stride_iota_twenty_two = std::views::stride(iota_twenty_two, 3);
    static_assert(std::ranges::sized_range<decltype(stride_iota_twenty_two)>); // expected-no-diagnostics
    static_assert(8 == stride_iota_twenty_two.size(),
                  "Striding by 3 through a 22 member list has size 8."); // expected-no-diagnostics
  }
  return true;
}

int main(int, char**) {
  static_assert(test());

  return 0;
}
