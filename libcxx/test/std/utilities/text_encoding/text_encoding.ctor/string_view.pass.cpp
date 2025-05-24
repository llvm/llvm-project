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

// text_encoding::text_encoding(string_view) noexcept

#include "test_macros.h"
#include "test_text_encoding.h"
#include <cassert>
#include <cstring>
#include <string_view>
#include <text_encoding>
#include <type_traits>

constexpr void test_ctor(std::string_view str, std::string_view expect, std::text_encoding::id expect_id) {
  auto te = std::text_encoding(str);
  assert(te.mib() == expect_id);
  assert(expect.compare(te.name()) == 0);
}

void test_correct_encoding_spellings() {
  for (auto pair : unique_encoding_data) {
    test_ctor(pair.name, pair.name, std::text_encoding::id{pair.mib});
  }
}

int main() {
  {
    static_assert(std::is_nothrow_constructible<std::text_encoding, std::string_view>::value,
                  "Must be nothrow constructible with string_view");
  }

  // happy paths
  {
    test_correct_encoding_spellings();
  }

  {
    test_ctor("U_T_F-8", "U_T_F-8", std::text_encoding::UTF8);
  }

  {
    test_ctor("utf8", "utf8", std::text_encoding::UTF8);
  }

  {
    test_ctor("u.t.f-008", "u.t.f-008", std::text_encoding::UTF8);
  }

  {
    test_ctor("utf-80", "utf-80", std::text_encoding::other);
  }

  {
    test_ctor("iso885931988", "iso885931988", std::text_encoding::ISOLatin3);
  }

  {
    test_ctor("iso00885931988", "iso00885931988", std::text_encoding::ISOLatin3);
  }
}
