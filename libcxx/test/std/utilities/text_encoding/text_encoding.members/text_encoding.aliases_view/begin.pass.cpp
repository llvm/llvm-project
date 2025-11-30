//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding::aliases_view::begin()

#include <cassert>
#include <ranges>
#include <text_encoding>

constexpr bool test() {
  // 1. begin() of an aliases_view from a single text_encoding object are the same.
  {
    std::text_encoding te = std::text_encoding(std::text_encoding::UTF8);

    std::text_encoding::aliases_view view1 = te.aliases();
    std::text_encoding::aliases_view view2 = te.aliases();

    assert(std::ranges::begin(view1) == std::ranges::begin(view2));
    assert(view1.begin() == view2.begin());
  }

  // 2. begin() of aliases_views of two text_encoding objects that represent the same ID but hold different names are the same.
  {
    std::text_encoding te1 = std::text_encoding("ANSI_X3.4-1968");
    std::text_encoding te2 = std::text_encoding("ANSI_X3.4-1986");

    std::text_encoding::aliases_view view1 = te1.aliases();
    std::text_encoding::aliases_view view2 = te2.aliases();

    assert(view1.begin() == view2.begin());
    assert(std::ranges::begin(view1) == std::ranges::begin(view2));
  }

  // 3. begin() of aliases_views of two text_encoding objects that represent different IDs are different.
  {
    std::text_encoding te1 = std::text_encoding(std::text_encoding::UTF8);
    std::text_encoding te2 = std::text_encoding(std::text_encoding::ASCII);

    std::text_encoding::aliases_view view1 = te1.aliases();
    std::text_encoding::aliases_view view2 = te2.aliases();

    assert(!(view1.begin() == view2.begin()));
    assert(!(std::ranges::begin(view1) == std::ranges::begin(view2)));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
