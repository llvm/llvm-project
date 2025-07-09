//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class as_input_view

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//  template<input_range V>
//    requires view<V>
//  class as_input_view : public view_interface<as_input_view<V>>

// Functional tests of std::ranges::as_input_view.

#include <algorithm>
#include <cassert>
#include <ranges>
#include <string>
#include <vector>

constexpr bool test() {
  std::vector<std::string> vec{"Hello", ",", " ", "World", "!"};
  std::string expectedStr = "Hello, World!";

  {
    auto view = vec | std::views::join;
    assert(std::ranges::equal(view, expectedStr));
  }
  { // Test as_input_view with a vector of strings.
    auto view = vec | std::views::as_input | std::views::join;
    assert(std::ranges::equal(view, expectedStr));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
