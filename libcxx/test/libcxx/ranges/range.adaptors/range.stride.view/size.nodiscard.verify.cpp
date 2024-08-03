//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test that appropriate (member) functions are properly marked no_discard

#include <ranges>

#include "../../../../std/ranges/range.adaptors/range.stride.view/types.h"

void test() {
  auto sv = std::views::stride(SimpleNoConstSizedCommonView(), 2);
  const auto const_sv = std::views::stride(SimpleCommonConstView(), 2);

  sv.size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  const_sv.size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
