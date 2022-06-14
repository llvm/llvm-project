//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string<charT,traits,Allocator>& assign(const charT* s); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(S s, const typename S::value_type* str, S expected)
{
    s.assign(str);
    LIBCPP_ASSERT(s.__invariants());
    assert(s == expected);
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    test(S(), "", S());
    test(S(), "12345", S("12345"));
    test(S(), "12345678901234567890", S("12345678901234567890"));

    test(S("12345"), "", S());
    test(S("12345"), "12345", S("12345"));
    test(S("12345"), "1234567890", S("1234567890"));

    test(S("12345678901234567890"), "", S());
    test(S("12345678901234567890"), "12345", S("12345"));
    test(S("12345678901234567890"), "12345678901234567890",
         S("12345678901234567890"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S(), "", S());
    test(S(), "12345", S("12345"));
    test(S(), "12345678901234567890", S("12345678901234567890"));

    test(S("12345"), "", S());
    test(S("12345"), "12345", S("12345"));
    test(S("12345"), "1234567890", S("1234567890"));

    test(S("12345678901234567890"), "", S());
    test(S("12345678901234567890"), "12345", S("12345"));
    test(S("12345678901234567890"), "12345678901234567890",
         S("12345678901234567890"));
  }
#endif

  { // test assignment to self
    typedef std::string S;
    S s_short = "123/";
    S s_long  = "Lorem ipsum dolor sit amet, consectetur/";

    s_short.assign(s_short.c_str());
    assert(s_short == "123/");
    s_short.assign(s_short.c_str() + 2);
    assert(s_short == "3/");

    s_long.assign(s_long.c_str() + 30);
    assert(s_long == "nsectetur/");
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
