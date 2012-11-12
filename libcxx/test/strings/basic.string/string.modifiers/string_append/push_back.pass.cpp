//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// void push_back(charT c)

#include <string>
#include <cassert>

template <class S>
void
test(S s, typename S::value_type c, S expected)
{
    s.push_back(c);
    assert(s.__invariants());
    assert(s == expected);
}

int main()
{
    typedef std::string S;
    test(S(), 'a', S(1, 'a'));
    test(S("12345"), 'a', S("12345a"));
    test(S("12345678901234567890"), 'a', S("12345678901234567890a"));
}
