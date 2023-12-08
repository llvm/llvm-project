//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr explicit sentinel(split_view& parent);

#include <cassert>
#include <ranges>
#include <type_traits>

#include "test_iterators.h"

// test explicit
using Range     = std::ranges::subrange<int*, sentinel_wrapper<int*>>;
using SplitView = std::ranges::split_view<Range, std::ranges::single_view<int>>;
using SplitSent = std::ranges::sentinel_t<SplitView>;

static_assert(std::is_constructible_v<SplitSent, SplitView&>);
static_assert(!std::is_convertible_v<SplitView&, SplitSent>);

constexpr bool test() {
  {
    int buffer[] = {0, 1, 2};
    Range input{buffer, sentinel_wrapper<int*>(buffer + 3)};
    SplitView sv(input, -1);
    auto it = sv.begin();

    SplitSent sent(sv);
    assert(sent != it);

    ++it;
    assert(sent == it);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
