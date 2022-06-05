//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: !stdlib=libc++ && (c++11 || c++14)

// <string_view>

// Test that <string_view> provides all of the arithmetic, enum, and pointer
// hash specializations.

#include <string_view>

#include "poisoned_hash_helper.h"

#include "test_macros.h"

int main(int, char**) {
  test_library_hash_specializations_available();
  {
    test_hash_enabled_for_type<std::string_view>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test_hash_enabled_for_type<std::wstring_view>();
#endif
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    test_hash_enabled_for_type<std::u8string_view>();
#endif
#ifndef TEST_HAS_NO_UNICODE_CHARS
    test_hash_enabled_for_type<std::u16string_view>();
    test_hash_enabled_for_type<std::u32string_view>();
#endif
  }

  return 0;
}
