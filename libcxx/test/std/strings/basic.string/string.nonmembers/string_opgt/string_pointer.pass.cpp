//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   bool operator>(const basic_string<charT,traits,Allocator>& lhs, const charT* rhs);

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(const S& lhs, const typename S::value_type* rhs, bool x)
{
    assert((lhs > rhs) == x);
}

int main(int, char**)
{
    {
    typedef std::string S;
    test(S(""), "", false);
    test(S(""), "abcde", false);
    test(S(""), "abcdefghij", false);
    test(S(""), "abcdefghijklmnopqrst", false);
    test(S("abcde"), "", true);
    test(S("abcde"), "abcde", false);
    test(S("abcde"), "abcdefghij", false);
    test(S("abcde"), "abcdefghijklmnopqrst", false);
    test(S("abcdefghij"), "", true);
    test(S("abcdefghij"), "abcde", true);
    test(S("abcdefghij"), "abcdefghij", false);
    test(S("abcdefghij"), "abcdefghijklmnopqrst", false);
    test(S("abcdefghijklmnopqrst"), "", true);
    test(S("abcdefghijklmnopqrst"), "abcde", true);
    test(S("abcdefghijklmnopqrst"), "abcdefghij", true);
    test(S("abcdefghijklmnopqrst"), "abcdefghijklmnopqrst", false);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S(""), "", false);
    test(S(""), "abcde", false);
    test(S(""), "abcdefghij", false);
    test(S(""), "abcdefghijklmnopqrst", false);
    test(S("abcde"), "", true);
    test(S("abcde"), "abcde", false);
    test(S("abcde"), "abcdefghij", false);
    test(S("abcde"), "abcdefghijklmnopqrst", false);
    test(S("abcdefghij"), "", true);
    test(S("abcdefghij"), "abcde", true);
    test(S("abcdefghij"), "abcdefghij", false);
    test(S("abcdefghij"), "abcdefghijklmnopqrst", false);
    test(S("abcdefghijklmnopqrst"), "", true);
    test(S("abcdefghijklmnopqrst"), "abcde", true);
    test(S("abcdefghijklmnopqrst"), "abcdefghij", true);
    test(S("abcdefghijklmnopqrst"), "abcdefghijklmnopqrst", false);
    }
#endif

  return 0;
}
