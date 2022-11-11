//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// Test that <string> provides all of the arithmetic, enum, and pointer
// hash specializations.

#include <string>

#include "constexpr_char_traits.h"
#include "poisoned_hash_helper.h"
#include "test_allocator.h"
#include "test_macros.h"

struct MyChar {
  char c;
};

int main(int, char**) {
  test_library_hash_specializations_available();
  {
    test_hash_enabled_for_type<std::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test_hash_enabled_for_type<std::wstring>();
#endif
#ifndef TEST_HAS_NO_CHAR8_T
    test_hash_enabled_for_type<std::u8string>();
#endif
    test_hash_enabled_for_type<std::u16string>();
    test_hash_enabled_for_type<std::u32string>();
    test_hash_enabled_for_type<std::basic_string<char, std::char_traits<char>, test_allocator<char>>>();
    test_hash_disabled_for_type<std::basic_string<MyChar, std::char_traits<MyChar>, std::allocator<MyChar>>>();
    test_hash_disabled_for_type<std::basic_string<char, constexpr_char_traits<char>, std::allocator<char>>>();
  }

  return 0;
}
