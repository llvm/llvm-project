//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding::text_encoding(id) noexcept

#include <algorithm>
#include <cassert>
#include <ranges>
#include <text_encoding>
#include <type_traits>

#include "../test_text_encoding.h"

using id = std::text_encoding::id;

constexpr void id_ctor(id i, id expect_id, std::string_view expect_name) {
  std::text_encoding te = std::text_encoding(i);

  assert(te.mib() == expect_id);
  assert(expect_name == te.name());
  assert(std::ranges::contains(te.aliases(), expect_name));
}

constexpr void id_ctors() {
  for (auto pair : unique_encoding_data) {
    id_ctor(id(pair.mib), id(pair.mib), pair.name);
  }
}

constexpr void test_unknown_other() {
  {
    std::text_encoding te = std::text_encoding(id::other);

    assert(te.mib() == id::other);
    assert(std::string_view("") == te.name());
    assert(std::ranges::empty(te.aliases()));
  }

  {
    std::text_encoding te = std::text_encoding(id::unknown);

    assert(te.mib() == id::unknown);
    assert(std::string_view("") == te.name());
    assert(std::ranges::empty(te.aliases()));
  }
}

constexpr bool test() {
  {
    // 2. Constructing an object with a valid id must set mib() and the name to the corresponding value.
    id_ctors();
  }

  {
    // 3. Constructing an object using id::unknown or id::other must set mib() to id::unknown or id::other, respectively, and the name to an empty string.
    test_unknown_other();
  }

  return true;
}

int main(int, char**) {
  {
    // 1. text_encoding(id) must be nothrow
    static_assert(std::is_nothrow_constructible<std::text_encoding, std::text_encoding::id>::value,
                  "Must be nothrow constructible with id");
  }

  {
    test();
    static_assert(test());
  }

  return 0;
}
