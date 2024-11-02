//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iota_view(iterator first, see below last);

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  {
    std::ranges::iota_view commonView(SomeInt(0), SomeInt(10));
    std::ranges::iota_view<SomeInt, SomeInt> io(commonView.begin(), commonView.end());
    assert(std::ranges::next(io.begin(), 10) == io.end());
  }

  {
    std::ranges::iota_view unreachableSent(SomeInt(0));
    std::ranges::iota_view<SomeInt> io(unreachableSent.begin(), std::unreachable_sentinel);
    assert(std::ranges::next(io.begin(), 10) != io.end());
  }

  {
    std::ranges::iota_view differentTypes(SomeInt(0), IntComparableWith(SomeInt(10)));
    std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>> io(differentTypes.begin(), differentTypes.end());
    assert(std::ranges::next(io.begin(), 10) == io.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

