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

// text_encoding::text_encoding(id) noexcept

// Concerns:
// 1. text_encoding(id) must be nothrow
// 2. Constructing an object with a valid id must set mib() and the name to the corresponding value.
// 3. Constructing an object using id::unknown must set mib() to id::unknown and the name to an empty string.
// 4. Constructing an object using id::other must set mib() to id::other and the name to an empty string.

#include "test_text_encoding.h"
#include <cassert>
#include <string_view>
#include <text_encoding>
#include <type_traits>

using te_id = std::text_encoding::id;

constexpr void test_ctor(te_id i, te_id expect_id, std::string_view expect_name) {
  auto te = std::text_encoding(i);
  assert(te.mib() == expect_id);
  assert(expect_name.compare(te.name()) == 0);
}

int main() {
  {
    static_assert(std::is_nothrow_constructible<std::text_encoding, std::text_encoding::id>::value,
                  "Must be nothrow constructible with id");
  }
  
  {
    for (auto pair : unique_encoding_data){
      test_ctor(te_id{pair.mib}, te_id{pair.mib}, pair.name);
    }
  }

  {
    for(int i = 2261; i < 2300; i++){ // test out of range id values
      test_ctor(te_id{i}, te_id::unknown, "");
    }
  }
}
