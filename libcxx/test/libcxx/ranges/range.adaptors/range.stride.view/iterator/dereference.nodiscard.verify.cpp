//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr decltype(auto) operator*() const {

#include <ranges>
#include <utility>

void test() {
  {
    int range[] = {1, 2, 3};
    auto view   = std::ranges::views::stride(range, 3);
    auto it     = view.begin();
    ++it;
    *std::as_const(it); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
}
