//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr explicit single_view(const T& t);
// constexpr explicit single_view(T&& t);

#include <cassert>
#include <ranges>
#include <utility>

#include "test_macros.h"

struct Empty {};
struct BigType { char buffer[64] = {10}; };

constexpr bool test() {
  {
    BigType bt;
    std::ranges::single_view<BigType> sv(bt);
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }
  {
    const BigType bt;
    const std::ranges::single_view<BigType> sv(bt);
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }

  {
    BigType bt;
    std::ranges::single_view<BigType> sv(std::move(bt));
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }
  {
    const BigType bt;
    const std::ranges::single_view<BigType> sv(std::move(bt));
    assert(sv.data()->buffer[0] == 10);
    assert(sv.size() == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
