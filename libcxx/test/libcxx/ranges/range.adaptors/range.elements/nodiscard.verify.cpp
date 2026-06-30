//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test the libc++ extension that std::ranges::transform_view and std::views::transform are marked as [[nodiscard]].

#include <ranges>
#include <utility>
#include <functional>
#include <string_view>
#include <map>

struct View : std::ranges::view_interface<View> {
  int* begin();
  const int* begin() const;
  volatile int* end();
  const volatile int* end() const;
};
static_assert(!std::ranges::common_range<View>);
static_assert(!std::same_as<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
static_assert(!std::same_as<std::ranges::sentinel_t<View>, std::ranges::sentinel_t<const View>>);

void test() {
    auto historical_figures = std::map{
        std::pair{"Lovelace", 1815},
        std::pair{"Turing", 1912},
        std::pair{"Babbage", 1791},
        std::pair{"Hamilton", 1936}
    };

    // [range.elements.overview]
    
   // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::elements<0>(historical_figures);

   // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::views::elements<1>(historical_figures);
}

