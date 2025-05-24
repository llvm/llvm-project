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

// text_encoding operator==(const text_encoding&, const text_encoding&) _NOEXCEPT 

// Concerns:
// 1. operator==(const text_encoding&, const text_encoding&) must be noexcept
// 2. operator==(const text_encoding&, const text_encoding&) returns true if both text_encoding ids are equal
// 3. operator==(const text_encoding&, const text_encoding&) for text_encodings with ids of "other" return true if the names are equal
// 4. operator==(const text_encoding&, const text_encoding&) returns false when comparingtext_encodings with different ids
// 5. operator==(const text_encoding&, const text_encoding&) for text_encodings with ids of "other" returns false if the names are not equal

#include <cassert>
#include <text_encoding>
#include <type_traits>

#include "test_macros.h"
#include "test_text_encoding.h"

using id = std::text_encoding::id;

int main(){
  
  { // 1
    auto te1 = std::text_encoding();
    auto te2 = std::text_encoding();
    ASSERT_NOEXCEPT(te1 == te2);
  }

  { // 2
    auto te1 = std::text_encoding(id::UTF8);
    auto te2 = std::text_encoding(id::UTF8);
    assert(te1 == te2);
  }

  { // 3
    auto other_te1 = std::text_encoding("foo");
    auto other_te2 = std::text_encoding("foo");
    assert(other_te1 == other_te2);
  }

  { // 4
    auto te1 = std::text_encoding(id::UTF8);
    auto te2 = std::text_encoding(id::UTF16);
    assert(!(te1 == te2));
  }

  { // 5
    auto other_te1 = std::text_encoding("foo");
    auto other_te2 = std::text_encoding("bar");
    assert(!(other_te1 == other_te2));
  }
}
