//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding::text_encoding(std::text_encoding::id) noexcept

#include <algorithm>
#include <cassert>
#include <concepts>
#include <ranges>
#include <text_encoding>
#include <type_traits>

#include "../test_text_encoding.h"

using id = std::text_encoding::id;

constexpr bool test() {
  {
    // 2. Constructing an object with a valid id must set mib() and the name to the corresponding value.
    for (auto pair : unique_encoding_data) {
      std::same_as<std::text_encoding> decltype(auto) te = std::text_encoding(id(pair.mib));

      assert(te.mib() == id(pair.mib));
      assert(pair.name == te.name());
      assert(std::ranges::contains(te.aliases(), pair.name));
    }
  }

  {
    // 3. Constructing an object using id::unknown or id::other must set mib() to id::unknown or id::other, respectively, and the name to an empty string.
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
