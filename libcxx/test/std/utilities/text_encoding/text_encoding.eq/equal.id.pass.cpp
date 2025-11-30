//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// bool text_encoding::operator==(const text_encoding&, id) noexcept

#include <cassert>
#include <text_encoding>

#include "test_macros.h"
#include "../test_text_encoding.h"

using id = std::text_encoding::id;

constexpr void test_primary_encodings() {
  for (auto& data : unique_encoding_data) {
    std::text_encoding te = std::text_encoding(id(data.mib));
    assert(te == id(data.mib));
  }
}

constexpr bool test() {
  // 1. operator==(const text_encoding&, id) must be noexcept
  {
    std::text_encoding te = std::text_encoding();
    ASSERT_NOEXCEPT(te == id::UTF8);
  }

  // 2. operator==(const text_encoding&, id) returns true if mib() is equal to the id
  {
    assert(std::text_encoding(id::UTF8) == id::UTF8);
  }

  // 2.1
  {
    assert(std::text_encoding() == id::unknown);
  }

  // 2.1.1
  {
    assert(std::text_encoding(id::unknown) == id::unknown);
  }

  // 2.2
  {
    assert(std::text_encoding(id::other) == id::other);
  }

  // 3. operator==(const text_encoding&, id) returns false if mib() is not equal to the id
  {
    assert(!(std::text_encoding(id::UTF8) == id::UTF16));
  }

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
