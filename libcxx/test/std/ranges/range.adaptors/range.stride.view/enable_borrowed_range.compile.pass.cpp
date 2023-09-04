//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include "test.h"
#include <ranges>

constexpr bool test() {
  using std::ranges::enable_borrowed_range;
  // Make sure that a stride_view over neither a borrowable nor an unborrowable view
  // is itself borrowable.
  static_assert(!enable_borrowed_range<std::ranges::stride_view<InstrumentedBasicView<int>>>);
  static_assert(!enable_borrowed_range<std::ranges::stride_view<InstrumentedBorrowedRange<int>>>);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
