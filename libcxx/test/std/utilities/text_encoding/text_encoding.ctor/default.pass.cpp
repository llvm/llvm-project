//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding::text_encoding() noexcept

#include <cassert>
#include <text_encoding>
#include <type_traits>

constexpr bool test() {
  std::text_encoding te = std::text_encoding();
  assert(te.mib() == std::text_encoding::unknown);
  assert(std::string_view("") == te.name());

  return true;
}

int main(int, char**) {
  // 1. Default constructor must be nothrow
  {
    static_assert(
        std::is_nothrow_default_constructible<std::text_encoding>::value, "Must be nothrow default constructible");
  }

  // 2. Default constructing a text_encoding object makes it so that mib() == id::unknown, and its name is empty
  {
    test();
    static_assert(test());
  }

  return 0;
}
