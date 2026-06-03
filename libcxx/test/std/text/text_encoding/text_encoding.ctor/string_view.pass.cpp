//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=40000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=1000000000

// <text_encoding>

// text_encoding::text_encoding(string_view) noexcept

#include <cassert>
#include <concepts>
#include <string_view>
#include <text_encoding>
#include <type_traits>

#include "../test_text_encoding.h"

constexpr void test_ctor(std::string_view str, std::text_encoding::id expect_id) {
  std::same_as<std::text_encoding> decltype(auto) te = std::text_encoding(str);

  assert(te.mib() == expect_id);
  assert(te.name() == str);
}

constexpr bool test() {
  { // The first encoding name for each mib in the data table.
    for (auto& data : unique_encoding_data) {
      std::same_as<std::text_encoding> decltype(auto) te = std::text_encoding(data.name);

      assert(te.mib() == std::text_encoding::id(data.mib));
      assert(te.name() == data.name);
    }
  }

  { // Names that should all result in an "other" text encoding
    for (auto& name : other_names) {
      std::same_as<std::text_encoding> decltype(auto) te = std::text_encoding(name);

      assert(te.mib() == std::text_encoding::other);
      assert(te.name() == name);
    }
  }

  test_ctor("U_T_F-8", std::text_encoding::UTF8);
  test_ctor("utf8", std::text_encoding::UTF8);
  test_ctor("u.t.f-008", std::text_encoding::UTF8);
  test_ctor("utf-80", std::text_encoding::other);
  test_ctor("iso885931988", std::text_encoding::ISOLatin3);
  test_ctor("iso00885931988", std::text_encoding::ISOLatin3);

  return true;
}

int main(int, char**) {
  {
    static_assert(std::is_nothrow_constructible_v<std::text_encoding, std::string_view>,
                  "Must be nothrow constructible with string_view");
  }

  {
    test();
    static_assert(test());
  }

  {
    // Runtime test only as it would take unreasonably long to test in constexpr.
    for (auto& enc : all_encoding_data) {
      test_ctor(enc.name, std::text_encoding::id(enc.mib));
    }
  }

  return 0;
}
