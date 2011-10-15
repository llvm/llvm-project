//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string<charT,traits,Allocator>& operator+=(charT c);

#include <string>
#include <cassert>

template <class S>
void
test(S s, typename S::value_type str, S expected)
{
    s += str;
    assert(s.__invariants());
    assert(s == expected);
}

int main()
{
    typedef std::string S;
    test(S(), 'a', S("a"));
    test(S("12345"), 'a', S("12345a"));
    test(S("1234567890"), 'a', S("1234567890a"));
    test(S("12345678901234567890"), 'a', S("12345678901234567890a"));
}
