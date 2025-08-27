//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=9000000

// struct text_encoding::aliases_view

// Concerns:
// 1. An alias_view of a text_encoding object for "other" is empty
// 2. An alias_view of a text_encoding object for "unknown" is empty
// 3. An alias_view of a text_encoding object for a known encoding e.g. "UTF-8" is not empty

#include "test_text_encoding.h"

using id = std::text_encoding::id;

constexpr bool test_other() {
  auto te1         = std::text_encoding(id::other);
  auto empty_range = te1.aliases();
  assert(std::ranges::empty(empty_range) && empty_range.empty() && !bool(empty_range));

  for (auto& other_name : other_names) {
    auto te_other          = std::text_encoding(other_name);
    auto empty_range_other = te_other.aliases();
    assert(std::ranges::empty(empty_range_other) && empty_range_other.empty() && !bool(empty_range_other));
  }

  return true;
}

constexpr bool test_unknown() {
  auto te          = std::text_encoding(id::unknown);
  auto empty_range = te.aliases();
  return std::ranges::empty(empty_range) && empty_range.empty() && !bool(empty_range);
}

constexpr bool test_primary_encodings() {
  for (auto& data : unique_encoding_data) {
    auto te    = std::text_encoding(id(data.mib));
    auto range = te.aliases();

    assert(!std::ranges::empty(range));
    assert(!range.empty());
    assert(bool(range));
  }
  return true;
}

int main() {
  {
    static_assert(test_other());
    assert(test_other());
  }

  {
    static_assert(test_unknown());
    assert(test_unknown());
  }

  {
    static_assert(test_primary_encodings());
    assert(test_primary_encodings());
  }
}
