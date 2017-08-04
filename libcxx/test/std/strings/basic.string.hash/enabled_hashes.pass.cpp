//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <string>

// Test that <string> provides all of the arithmetic, enum, and pointer
// hash specializations.

#include <string>

#include "poisoned_hash_helper.hpp"

int main() {
  test_library_hash_specializations_available();
  {
    test_hash_enabled_for_type<std::string>();
    test_hash_enabled_for_type<std::wstring>();
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test_hash_enabled_for_type<std::u16string>();
    test_hash_enabled_for_type<std::u32string>();
#endif
  }
}
