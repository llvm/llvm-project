//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// std::ranges::lazy_split_view::outer-iterator::value_type::value_type()

#include <ranges>

#include "../types.h"

constexpr bool test() {
  {
    [[maybe_unused]] ValueTypeForward val;
  }

  {
    [[maybe_unused]] ValueTypeForward val = {};
  }

  {
    [[maybe_unused]] ValueTypeInput val;
  }

  {
    [[maybe_unused]] ValueTypeInput val = {};
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
