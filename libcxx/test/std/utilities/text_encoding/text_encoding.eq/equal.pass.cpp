//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// bool text_encoding::operator==(const text_encoding&, const text_encoding&) noexcept

#include <cassert>
#include <text_encoding>

#include "test_macros.h"

using id = std::text_encoding::id;

constexpr bool test() {
  // 1. operator==(const text_encoding&, const text_encoding&) must be noexcept
  {
    ASSERT_NOEXCEPT(std::text_encoding() == std::text_encoding());
  }

  // 2. operator==(const text_encoding&, const text_encoding&) returns true if both text_encoding ids are equal
  {
    std::text_encoding te1 = std::text_encoding(id::UTF8);
    std::text_encoding te2 = std::text_encoding(id::UTF8);
    assert(te1 == te2);
  }

  // 3. operator==(const text_encoding&, const text_encoding&) for text_encodings with ids of "other" return true if the names are equal
  {
    std::text_encoding other_te1 = std::text_encoding("foo");
    std::text_encoding other_te2 = std::text_encoding("foo");
    assert(other_te1 == other_te2);
  }

  // 4. operator==(const text_encoding&, const text_encoding&) returns false when comparingtext_encodings with different ids
  {
    std::text_encoding te1 = std::text_encoding(id::UTF8);
    std::text_encoding te2 = std::text_encoding(id::UTF16);
    assert(!(te1 == te2));
  }

  // 5. operator==(const text_encoding&, const text_encoding&) for text_encodings with ids of "other" returns false if the names are not equal
  {
    std::text_encoding other_te1 = std::text_encoding("foo");
    std::text_encoding other_te2 = std::text_encoding("bar");
    assert(!(other_te1 == other_te2));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
