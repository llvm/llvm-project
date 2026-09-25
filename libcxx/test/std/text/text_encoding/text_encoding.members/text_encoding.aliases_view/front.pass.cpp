//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding::aliases_view::front()

#include <cassert>
#include <ranges>
#include <string_view>
#include <text_encoding>

using id = std::text_encoding::id;

constexpr bool test() {
  // 1. An aliases_view from a single text_encoding object returns the same front()
  {
    std::text_encoding te{id::UTF8};

    auto view1 = te.aliases();
    auto view2 = te.aliases();

    assert(std::string_view(view1.front()) == std::string_view(view2.front()));
  }

  // 2. An aliases_views of two text_encoding objects that represent the same ID but hold different names return the same front()
  {
    std::text_encoding te1{"US-ASCII"};
    std::text_encoding te2{"ANSI_X3.4-1986"};

    auto view1 = te1.aliases();
    auto view2 = te2.aliases();

    assert(std::string_view(view1.front()) == std::string_view(view2.front()));
  }

  // 3. An aliases_views of two text_encoding objects that represent different IDs return different front()
  {
    std::text_encoding te1{id::UTF8};
    std::text_encoding te2{id::ASCII};

    auto view1 = te1.aliases();
    auto view2 = te2.aliases();

    assert(!(std::string_view(view1.front()) == std::string_view(view2.front())));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
