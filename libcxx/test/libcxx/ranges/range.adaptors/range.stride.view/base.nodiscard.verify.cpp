//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test that std::ranges::stride_view::base() is marked nodiscard.

#include <ranges>

#include "../../../../std/ranges/range.adaptors/range.stride.view/types.h"

void test() {
  const std::vector<int> intv = {1, 2, 3};
  auto copyable_view          = CopyableView<std::vector<int>::const_iterator>(intv.begin(), intv.end());

  static_assert(std::copy_constructible<decltype(copyable_view)>);

  const auto sv = std::ranges::stride_view<decltype(copyable_view)>(copyable_view, 2);

  sv.base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::move(sv).base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
