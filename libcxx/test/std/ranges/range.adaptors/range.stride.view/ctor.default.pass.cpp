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
#include <type_traits>

constexpr bool test() {
  // There is no default ctor for stride_view.
  static_assert(!std::is_default_constructible_v<std::ranges::stride_view<BidirView>>);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
