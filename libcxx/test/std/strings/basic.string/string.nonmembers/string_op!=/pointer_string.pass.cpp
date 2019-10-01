//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   bool operator!=(const charT* lhs, const basic_string<charT,traits,Allocator>& rhs);

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(const typename S::value_type* lhs, const S& rhs, bool x)
{
    assert((lhs != rhs) == x);
}

int main(int, char**)
{
    {
    typedef std::string S;
    test("", S(""), false);
    test("", S("abcde"), true);
    test("", S("abcdefghij"), true);
    test("", S("abcdefghijklmnopqrst"), true);
    test("abcde", S(""), true);
    test("abcde", S("abcde"), false);
    test("abcde", S("abcdefghij"), true);
    test("abcde", S("abcdefghijklmnopqrst"), true);
    test("abcdefghij", S(""), true);
    test("abcdefghij", S("abcde"), true);
    test("abcdefghij", S("abcdefghij"), false);
    test("abcdefghij", S("abcdefghijklmnopqrst"), true);
    test("abcdefghijklmnopqrst", S(""), true);
    test("abcdefghijklmnopqrst", S("abcde"), true);
    test("abcdefghijklmnopqrst", S("abcdefghij"), true);
    test("abcdefghijklmnopqrst", S("abcdefghijklmnopqrst"), false);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test("", S(""), false);
    test("", S("abcde"), true);
    test("", S("abcdefghij"), true);
    test("", S("abcdefghijklmnopqrst"), true);
    test("abcde", S(""), true);
    test("abcde", S("abcde"), false);
    test("abcde", S("abcdefghij"), true);
    test("abcde", S("abcdefghijklmnopqrst"), true);
    test("abcdefghij", S(""), true);
    test("abcdefghij", S("abcde"), true);
    test("abcdefghij", S("abcdefghij"), false);
    test("abcdefghij", S("abcdefghijklmnopqrst"), true);
    test("abcdefghijklmnopqrst", S(""), true);
    test("abcdefghijklmnopqrst", S("abcde"), true);
    test("abcdefghijklmnopqrst", S("abcdefghij"), true);
    test("abcdefghijklmnopqrst", S("abcdefghijklmnopqrst"), false);
    }
#endif

  return 0;
}
