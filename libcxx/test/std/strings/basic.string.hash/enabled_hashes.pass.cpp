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

template <>
struct std::char_traits<MyChar> {
  using char_type  = MyChar;
  using int_type   = std::char_traits<char>::int_type;
  using off_type   = std::char_traits<char>::off_type;
  using pos_type   = std::char_traits<char>::pos_type;
  using state_type = std::char_traits<char>::state_type;

  static void assign(char_type&, const char_type&);
  static bool eq(char_type, char_type);
  static bool lt(char_type, char_type);

  static int              compare(const char_type*, const char_type*, std::size_t);
  static std::size_t           length(const char_type*);
  static const char_type* find(const char_type*, std::size_t, const char_type&);
  static char_type*       move(char_type*, const char_type*, std::size_t);
  static char_type*       copy(char_type*, const char_type*, std::size_t);
  static char_type*       assign(char_type*, std::size_t, char_type);

  static int_type  not_eof(int_type);
  static char_type to_char_type(int_type);
  static int_type  to_int_type(char_type);
  static bool      eq_int_type(int_type, int_type);
  static int_type  eof();
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
