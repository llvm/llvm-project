//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=9000000

// <text_encoding>

// text_encoding::aliases_view

#include <cassert>
#include <ranges>
#include <text_encoding>

#include "../../test_text_encoding.h"

using id = std::text_encoding::id;

constexpr void test_other_unknown() {
  {
    std::text_encoding te_other = std::text_encoding(id::other);

    std::text_encoding::aliases_view other_range = te_other.aliases();

    assert(std::ranges::empty(other_range));
    assert(other_range.empty());
    assert(!bool(other_range));
  }

  {
    std::text_encoding te_unknown = std::text_encoding(id::unknown);

    std::text_encoding::aliases_view unknown_range = te_unknown.aliases();
    assert(std::ranges::empty(unknown_range));
    assert(unknown_range.empty());
    assert(!bool(unknown_range));
  }
}

constexpr void test_primary_encodings() {
  for (auto& data : unique_encoding_data) {
    std::text_encoding te = std::text_encoding(id(data.mib));

    std::text_encoding::aliases_view range = te.aliases();

    assert(!std::ranges::empty(range));
    assert(!range.empty());
    assert(bool(range));
  }
}

constexpr bool test() {
  // 1. An alias_view of a text_encoding object for "other" and "unknown" are empty
  {
    test_other_unknown();
  }

  // 2. An alias_view of a text_encoding object for a known encoding e.g. "UTF-8" is not empty
  {
    test_primary_encodings();
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
