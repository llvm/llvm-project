//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// view_interface<text_encoding::aliases_view>::operator bool()

#include <cassert>
#include <ranges>
#include <text_encoding>

#include "../../test_text_encoding.h"

using id = std::text_encoding::id;

constexpr bool test() {
  // 1. An alias_view of a text_encoding object for "other" and "unknown" are empty
  {
    {
      std::text_encoding te_other{id::other};
      auto other_range = te_other.aliases();
      assert(!bool(other_range));
    }

    {
      std::text_encoding te_unknown{id::unknown};
      auto unknown_range = te_unknown.aliases();
      assert(!bool(unknown_range));
    }
  }

  // 2. An alias_view of a text_encoding object for a known encoding e.g. "UTF-8" is not empty
  {
    for (auto& data : unique_encoding_data) {
      std::text_encoding te{id(data.mib)};
      auto range = te.aliases();
      assert(bool(range));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
