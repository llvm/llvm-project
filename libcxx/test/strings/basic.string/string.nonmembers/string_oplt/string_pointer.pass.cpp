//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   bool operator<(const basic_string<charT,traits,Allocator>& lhs, const charT* rhs);

#include <string>
#include <cassert>

template <class S>
void
test(const S& lhs, const typename S::value_type* rhs, bool x)
{
    assert((lhs < rhs) == x);
}

typedef std::string S;

int main()
{
    test(S(""), "", false);
    test(S(""), "abcde", true);
    test(S(""), "abcdefghij", true);
    test(S(""), "abcdefghijklmnopqrst", true);
    test(S("abcde"), "", false);
    test(S("abcde"), "abcde", false);
    test(S("abcde"), "abcdefghij", true);
    test(S("abcde"), "abcdefghijklmnopqrst", true);
    test(S("abcdefghij"), "", false);
    test(S("abcdefghij"), "abcde", false);
    test(S("abcdefghij"), "abcdefghij", false);
    test(S("abcdefghij"), "abcdefghijklmnopqrst", true);
    test(S("abcdefghijklmnopqrst"), "", false);
    test(S("abcdefghijklmnopqrst"), "abcde", false);
    test(S("abcdefghijklmnopqrst"), "abcdefghij", false);
    test(S("abcdefghijklmnopqrst"), "abcdefghijklmnopqrst", false);
}
