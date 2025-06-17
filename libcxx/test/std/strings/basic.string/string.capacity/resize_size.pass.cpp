//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void resize(size_type n); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <cassert>

#include "asan_testing.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_macros.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s, typename S::size_type n, S expected) {
  s.resize(n);
  LIBCPP_ASSERT(s.__invariants());
  assert(s == expected);
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class CharT, class Alloc>
TEST_CONSTEXPR_CXX20 void test_string() {
  {
    using string_type = std::basic_string<CharT, std::char_traits<CharT>, Alloc>;
    test(string_type(), 0, string_type());
    test(string_type(), 1, string_type(1, '\0'));
    test(string_type(), 10, string_type(10, '\0'));
    test(string_type(), 100, string_type(100, '\0'));
    test(string_type(MAKE_CSTRING(CharT, "12345")), 0, string_type());
    test(string_type(MAKE_CSTRING(CharT, "12345")), 2, string_type(MAKE_CSTRING(CharT, "12")));
    test(string_type(MAKE_CSTRING(CharT, "12345")), 5, string_type(MAKE_CSTRING(CharT, "12345")));
    test(string_type(MAKE_CSTRING(CharT, "12345")),
         15,
         string_type(MAKE_CSTRING(CharT, "12345\0\0\0\0\0\0\0\0\0\0"), 15));
    test(string_type(MAKE_CSTRING(CharT, "12345678901234567890123456789012345678901234567890")), 0, string_type());
    test(string_type(MAKE_CSTRING(CharT, "12345678901234567890123456789012345678901234567890")),
         10,
         string_type(MAKE_CSTRING(CharT, "1234567890")));
    test(string_type(MAKE_CSTRING(CharT, "12345678901234567890123456789012345678901234567890")),
         50,
         string_type(MAKE_CSTRING(CharT, "12345678901234567890123456789012345678901234567890")));
    test(
        string_type(MAKE_CSTRING(CharT, "12345678901234567890123456789012345678901234567890")),
        60,
        string_type(MAKE_CSTRING(CharT, "12345678901234567890123456789012345678901234567890\0\0\0\0\0\0\0\0\0\0"), 60));
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    std::basic_string<CharT, std::char_traits<CharT>, Alloc> str;
    try {
      str.resize(std::string::npos);
      assert(false);
    } catch (const std::length_error&) {
    }
  }
#endif

  { // check that string can grow to max_size()
    std::basic_string<CharT, std::char_traits<CharT>, tiny_size_allocator<32, CharT> > str;
    str.resize(str.max_size());
    assert(str.size() == str.max_size());
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<char, std::allocator<char> >();
#if TEST_STD_VER >= 11
  test_string<char, min_allocator<char>>();
  test_string<char, safe_allocator<char>>();
#endif

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_string<wchar_t, std::allocator<wchar_t> >();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
