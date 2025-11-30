//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=30000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=99000000

// <text_encoding>

// text_encoding::text_encoding(string_view) noexcept

#include <cassert>
#include <string_view>
#include <text_encoding>
#include <type_traits>

#include "../test_text_encoding.h"

constexpr void test_ctor(std::string_view str, std::string_view expect, std::text_encoding::id expect_id) {
  std::text_encoding te = std::text_encoding(str);

  assert(te.mib() == expect_id);
  assert(te.name() == expect);
}

constexpr void test_primary_encoding_spellings() {
  for (auto& data : unique_encoding_data) {
    std::text_encoding te = std::text_encoding(data.name);

    assert(te.mib() == std::text_encoding::id(data.mib));
    assert(te.name() == data.name);
  }
}

constexpr void test_others() {
  for (auto& name : other_names) {
    std::text_encoding te = std::text_encoding(name);

    assert(te.mib() == std::text_encoding::other);
    assert(te.name() == name);
  }
}

constexpr bool test() {
  // happy paths
  {
    test_primary_encoding_spellings();
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

  {
    test_others();
  }

  return true;
}

int main(int, char**) {
  {
    static_assert(std::is_nothrow_constructible<std::text_encoding, std::string_view>::value,
                  "Must be nothrow constructible with string_view");
  }

  {
    test();
    static_assert(test());
  }

  return 0;
}
