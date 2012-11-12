//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<class InputIterator>
//   iterator insert(const_iterator p, InputIterator first, InputIterator last);

#include <string>
#include <cassert>

#include "../../input_iterator.h"

template <class S, class It>
void
test(S s, typename S::difference_type pos, It first, It last, S expected)
{
    typename S::const_iterator p = s.cbegin() + pos;
    typename S::iterator i = s.insert(p, first, last);
    assert(s.__invariants());
    assert(i - s.begin() == pos);
    assert(s == expected);
}

int main()
{
    typedef std::string S;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test(S(), 0, s, s, S());
    test(S(), 0, s, s+1, S("A"));
    test(S(), 0, s, s+10, S("ABCDEFGHIJ"));
    test(S(), 0, s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), 0, s, s, S("12345"));
    test(S("12345"), 1, s, s+1, S("1A2345"));
    test(S("12345"), 4, s, s+10, S("1234ABCDEFGHIJ5"));
    test(S("12345"), 5, s, s+52, S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), 0, s, s, S("1234567890"));
    test(S("1234567890"), 1, s, s+1, S("1A234567890"));
    test(S("1234567890"), 10, s, s+10, S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), 8, s, s+52, S("12345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz90"));

    test(S("12345678901234567890"), 3, s, s, S("12345678901234567890"));
    test(S("12345678901234567890"), 3, s, s+1, S("123A45678901234567890"));
    test(S("12345678901234567890"), 15, s, s+10, S("123456789012345ABCDEFGHIJ67890"));
    test(S("12345678901234567890"), 20, s, s+52,
         S("12345678901234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S());
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("A"));
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("ABCDEFGHIJ"));
    test(S(), 0, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S("12345"));
    test(S("12345"), 1, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("1A2345"));
    test(S("12345"), 4, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("1234ABCDEFGHIJ5"));
    test(S("12345"), 5, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), 0, input_iterator<const char*>(s), input_iterator<const char*>(s), S("1234567890"));
    test(S("1234567890"), 1, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("1A234567890"));
    test(S("1234567890"), 10, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), 8, input_iterator<const char*>(s), input_iterator<const char*>(s+52), S("12345678ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz90"));

    test(S("12345678901234567890"), 3, input_iterator<const char*>(s), input_iterator<const char*>(s), S("12345678901234567890"));
    test(S("12345678901234567890"), 3, input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("123A45678901234567890"));
    test(S("12345678901234567890"), 15, input_iterator<const char*>(s), input_iterator<const char*>(s+10), S("123456789012345ABCDEFGHIJ67890"));
    test(S("12345678901234567890"), 20, input_iterator<const char*>(s), input_iterator<const char*>(s+52),
         S("12345678901234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
}
