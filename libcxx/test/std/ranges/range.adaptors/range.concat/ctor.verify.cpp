//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <ranges>
#include <string>
#include <vector>

#include "test_macros.h"

template <typename... Ts>
concept ViewWithAtLeastOneElement = requires(Ts...&& a) { std::views::concat(a...); };

int main(int, char**) {
  {
    // LWG 4082
    std::vector<int> v{1, 2, 3};
    auto r = std::views::counted(std::back_inserter(v), 3);
    auto c = std::views::concat(r);
    // expected-error@*:* {{}}
  }

  {
    // input is not a view
    int x  = 1;
    auto c = std::views::concat(x);
    // expected-error@*:* {{}}
  }

  {
    // input is a view and has 0 size
    static_assert(!ViewWithAtLeastOneElement<>);
  }

  {
    // input is a view and has at least an element
    static_assert(ViewWithAtLeastOneElement<std::vector<int>>);
  }

  {
    // inputs are non-concatable
    std::vector<int> v1{1, 2};
    std::vector<std::string> v2{"Hello", "World"};
    auto c = std::views::concat(v1, v2);
    // expected-error@*:* {{}}
  }

  return 0;
}
