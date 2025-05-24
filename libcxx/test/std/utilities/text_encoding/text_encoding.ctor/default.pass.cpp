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

// text_encoding::text_encoding() noexcept

// Concerns:
// 1. Default constructor must be nothrow
// 2. Default constructing a text_encoding object makes it so that mib() == id::unknown, and its name is empty

#include <cassert>
#include <cstring>
#include <text_encoding>
#include <type_traits>

int main(int, char**) {
  {
    static_assert(
        std::is_nothrow_default_constructible<std::text_encoding>::value, "Must be nothrow default constructible");
  }

  {
    auto te = std::text_encoding();
    assert(te.mib() == std::text_encoding::id::unknown);
    assert(strcmp(te.name(), "") == 0);
  }
}
