//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr InnerIter operator->() const
//   requires has-arrow<InnerIter> && copyable<InnerIter>;

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  Box buffer[4][4] = {{{1111}, {2222}, {3333}, {4444}}, {{555}, {666}, {777}, {888}}, {{99}, {1010}, {1111}, {1212}}, {{13}, {14}, {15}, {16}}};

  {
    // Copyable input iterator with arrow.
    ValueView<Box> children[4] = {ValueView(buffer[0]), ValueView(buffer[1]), ValueView(buffer[2]), ValueView(buffer[3])};
    std::ranges::join_view jv(ValueView<ValueView<Box>>{children});
    assert(jv.begin()->x == 1111);
  }

  {
    std::ranges::join_view jv(buffer);
    assert(jv.begin()->x == 1111);
  }

  {
    const std::ranges::join_view jv(buffer);
    assert(jv.begin()->x == 1111);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
