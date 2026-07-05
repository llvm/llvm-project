//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// sentinel() = default;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  using Sent = std::ranges::sentinel_t<std::ranges::iota_view<Int42<DefaultTo42>, IntComparableWith<Int42<DefaultTo42>>>>;
  using Iter = std::ranges::iterator_t<std::ranges::iota_view<Int42<DefaultTo42>, IntComparableWith<Int42<DefaultTo42>>>>;
  assert(Sent() == Iter());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
