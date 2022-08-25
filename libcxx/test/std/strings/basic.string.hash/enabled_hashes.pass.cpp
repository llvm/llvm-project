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

#include "poisoned_hash_helper.h"

#include "test_macros.h"

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
  }

  return 0;
}
