//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string<charT,traits,Allocator>&
//   assign(const basic_string<charT,traits>& str); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "nasty_string.h"
#include "min_allocator.h"
#include "test_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S dest, S src) {
  dest.assign(src);
  LIBCPP_ASSERT(dest.__invariants());
  assert(dest == src);
}

template <class S>
TEST_CONSTEXPR_CXX20 void testAlloc(S dest, S src, const typename S::allocator_type& a) {
  dest.assign(src);
  LIBCPP_ASSERT(dest.__invariants());
  assert(dest == src);
  assert(dest.get_allocator() == a);
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_assign() {
  using CharT = typename S::value_type;

  test(S(), S());
  test(S(), S(CONVERT_TO_CSTRING(CharT, "12345")));
  test(S(), S(CONVERT_TO_CSTRING(CharT, "1234567890")));
  test(S(), S(CONVERT_TO_CSTRING(CharT, "12345678901234567890")));

  test(S(CONVERT_TO_CSTRING(CharT, "12345")), S());
  test(S(CONVERT_TO_CSTRING(CharT, "12345")), S(CONVERT_TO_CSTRING(CharT, "12345")));
  test(S(CONVERT_TO_CSTRING(CharT, "12345")), S(CONVERT_TO_CSTRING(CharT, "1234567890")));
  test(S(CONVERT_TO_CSTRING(CharT, "12345")), S(CONVERT_TO_CSTRING(CharT, "12345678901234567890")));

  test(S(CONVERT_TO_CSTRING(CharT, "1234567890")), S());
  test(S(CONVERT_TO_CSTRING(CharT, "1234567890")), S(CONVERT_TO_CSTRING(CharT, "12345")));
  test(S(CONVERT_TO_CSTRING(CharT, "1234567890")), S(CONVERT_TO_CSTRING(CharT, "1234567890")));
  test(S(CONVERT_TO_CSTRING(CharT, "1234567890")), S(CONVERT_TO_CSTRING(CharT, "12345678901234567890")));

  test(S(CONVERT_TO_CSTRING(CharT, "12345678901234567890")), S());
  test(S(CONVERT_TO_CSTRING(CharT, "12345678901234567890")), S(CONVERT_TO_CSTRING(CharT, "12345")));
  test(S(CONVERT_TO_CSTRING(CharT, "12345678901234567890")), S(CONVERT_TO_CSTRING(CharT, "1234567890")));
  test(S(CONVERT_TO_CSTRING(CharT, "12345678901234567890")), S(CONVERT_TO_CSTRING(CharT, "12345678901234567890")));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_assign<std::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_assign<std::wstring>();
#endif
#if TEST_STD_VER >= 20
  test_assign<std::u8string>();
#endif
#if TEST_STD_VER >= 11
  test_assign<std::u16string>();
  test_assign<std::u32string>();
#endif
#ifndef TEST_HAS_NO_NASTY_STRING
  test_assign<nasty_string>();
#endif

  {
    typedef std::string S;
    testAlloc(S(), S(), std::allocator<char>());
    testAlloc(S(), S("12345"), std::allocator<char>());
    testAlloc(S(), S("1234567890"), std::allocator<char>());
    testAlloc(S(), S("12345678901234567890"), std::allocator<char>());
  }

  {                                  //  LWG#5579 make sure assign takes the allocators where appropriate
    typedef other_allocator<char> A; // has POCCA --> true
    typedef std::basic_string<char, std::char_traits<char>, A> S;
    testAlloc(S(A(5)), S(A(3)), A(3));
    testAlloc(S(A(5)), S("1"), A());
    testAlloc(S(A(5)), S("1", A(7)), A(7));
    testAlloc(S(A(5)), S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), A(7));
    testAlloc(S("12345678901234567890", A(5)),
              S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)),
              A(7));
  }

#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test_assign<S>();
    testAlloc(S(), S(), min_allocator<char>());
    testAlloc(S(), S("12345"), min_allocator<char>());
    testAlloc(S(), S("1234567890"), min_allocator<char>());
    testAlloc(S(), S("12345678901234567890"), min_allocator<char>());
  }
#endif
#if TEST_STD_VER > 14
  {
    typedef std::string S;
    static_assert(noexcept(S().assign(S())), ""); // LWG#2063
  }
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
