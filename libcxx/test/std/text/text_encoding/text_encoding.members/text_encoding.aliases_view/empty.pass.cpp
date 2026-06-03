//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// view_interface<text_encoding::aliases_view>::empty()

#include <cassert>
#include <ranges>
#include <text_encoding>

#include "../../test_text_encoding.h"

using id = std::text_encoding::id;

constexpr bool test() {
  // 1. An alias_view of a text_encoding object for "other" and "unknown" are empty
  {
    {
      std::text_encoding te_other                  = std::text_encoding(id::other);
      std::text_encoding::aliases_view other_range = te_other.aliases();
      assert(other_range.empty());
    }

    {
      std::text_encoding te_unknown                  = std::text_encoding(id::unknown);
      std::text_encoding::aliases_view unknown_range = te_unknown.aliases();
      assert(unknown_range.empty());
    }
  }

  // 2. An alias_view of a text_encoding object for a known encoding e.g. "UTF-8" is not empty
  {
    for (auto& data : unique_encoding_data) {
      std::text_encoding te                  = std::text_encoding(id(data.mib));
      std::text_encoding::aliases_view range = te.aliases();
      assert(!range.empty());
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
