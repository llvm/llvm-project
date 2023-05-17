//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   bool operator==(const charT* lhs, const basic_string<charT,traits,Allocator>& rhs); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(const typename S::value_type* lhs, const S& rhs, bool x)
{
    assert((lhs == rhs) == x);
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    test("", S(""), true);
    test("", S("abcde"), false);
    test("", S("abcdefghij"), false);
    test("", S("abcdefghijklmnopqrst"), false);
    test("abcde", S(""), false);
    test("abcde", S("abcde"), true);
    test("abcde", S("abcdefghij"), false);
    test("abcde", S("abcdefghijklmnopqrst"), false);
    test("abcdefghij", S(""), false);
    test("abcdefghij", S("abcde"), false);
    test("abcdefghij", S("abcdefghij"), true);
    test("abcdefghij", S("abcdefghijklmnopqrst"), false);
    test("abcdefghijklmnopqrst", S(""), false);
    test("abcdefghijklmnopqrst", S("abcde"), false);
    test("abcdefghijklmnopqrst", S("abcdefghij"), false);
    test("abcdefghijklmnopqrst", S("abcdefghijklmnopqrst"), true);
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test("", S(""), true);
    test("", S("abcde"), false);
    test("", S("abcdefghij"), false);
    test("", S("abcdefghijklmnopqrst"), false);
    test("abcde", S(""), false);
    test("abcde", S("abcde"), true);
    test("abcde", S("abcdefghij"), false);
    test("abcde", S("abcdefghijklmnopqrst"), false);
    test("abcdefghij", S(""), false);
    test("abcdefghij", S("abcde"), false);
    test("abcdefghij", S("abcdefghij"), true);
    test("abcdefghij", S("abcdefghijklmnopqrst"), false);
    test("abcdefghijklmnopqrst", S(""), false);
    test("abcdefghijklmnopqrst", S("abcde"), false);
    test("abcdefghijklmnopqrst", S("abcdefghij"), false);
    test("abcdefghijklmnopqrst", S("abcdefghijklmnopqrst"), true);
  }
#endif

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
