//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26

// class text_encoding

// text_encoding::text_encoding() noexcept

// Concerns:
// 1. Default constructor must be nothrow
// 2. Default constructing a text_encoding object makes it so that mib() == id::unknown, and its name is empty

#include "test_text_encoding.h"

int main(int, char**) {
  {
    static_assert(
        std::is_nothrow_default_constructible<std::text_encoding>::value, "Must be nothrow default constructible");
  }

  {
    constexpr auto te = std::text_encoding();
    static_assert(te.mib() == std::text_encoding::id::unknown);
    static_assert(std::string_view("").compare(te.name()) == 0);
    assert(te.mib() == std::text_encoding::id::unknown);
    assert(std::string_view("").compare(te.name()) == 0);
  }
}
