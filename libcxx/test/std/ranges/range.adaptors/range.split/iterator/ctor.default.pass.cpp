//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// iterator() = default;

#include <cassert>
#include <ranges>

#include "../types.h"

struct PODIter : ForwardIterBase<PODIter> {
  int i;
};

constexpr bool test() {
  using SplitView = std::ranges::split_view<std::ranges::subrange<PODIter>, std::ranges::subrange<PODIter>>;
  using SplitIter = std::ranges::iterator_t<SplitView>;
  {
    SplitIter iter;
    assert(iter.base().i == 0); // PODIter has to be initialised to have value 0
  }

  {
    SplitIter iter = {};        // explicit(false)
    assert(iter.base().i == 0); // PODIter has to be initialised to have value 0
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
