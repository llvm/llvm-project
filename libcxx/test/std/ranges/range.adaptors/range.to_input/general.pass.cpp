//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class to_input_view

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class to_input_view

// Functional tests of std::ranges::to_input_view.

#include <cassert>
#include <ranges>
#include <string>
#include <vector>
#include <algorithm>

template <class R, class I>
constexpr bool isEqual(R& r, I i) {
  for (auto e : r)
    if (e != *i++)
      return false;

  return true;
}

constexpr bool test() {
  std::vector<std::string> vec{"Hello", ",", " ", "World", "!"};
  std::string expectedStr = "Hello, World!";

  {
    auto view = vec | std::views::join;
    assert(isEqual(view, expectedStr.begin()));
  }
  { // Test to_input_view with a vector of strings.
    auto view = vec | std::views::to_input | std::views::join;
    assert(isEqual(view, expectedStr.begin()));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
