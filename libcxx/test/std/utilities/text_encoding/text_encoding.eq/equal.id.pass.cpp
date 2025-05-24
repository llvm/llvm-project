//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26

// UNSUPPORTED: no-localization

// class text_encoding

// text_encoding operator==(const text_encoding&, id) _NOEXCEPT 

// Concerns:
// 1. operator==(const text_encoding&, id) must be noexcept
// 2. operator==(const text_encoding&, id) returns true if mib() is equal to the id
// 3. operator==(const text_encoding&, id) returns false if mib() is not equal to the id

#include <cassert>
#include <text_encoding>
#include <type_traits>

#include "test_macros.h"
#include "test_text_encoding.h"

using id = std::text_encoding::id;

int main() {

  { // 1
    auto te = std::text_encoding();
    ASSERT_NOEXCEPT(te == id::UTF8);
  }

  { // 2
    auto te = std::text_encoding(id::UTF8);
    assert(te == id::UTF8);
  }

  { // 2.0.1
    constexpr auto te = std::text_encoding();
    static_assert(te == id::unknown);
  }

  { // 2.1
    auto te = std::text_encoding(id::other);
    assert(te == id::other);
  }

  { // 2.1.1
    constexpr auto te = std::text_encoding(id::other);
    static_assert(te == id::other);
  }

  { // 3
    auto te = std::text_encoding(id::UTF8);
    assert(!(te == id::UTF16));
  }

  { // 3
    constexpr auto te = std::text_encoding(id::UTF8);
    static_assert(!(te == id::UTF16));
  }
}
