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
//   basic_string& append(InputIterator first, InputIterator last);

#include <string>
#include <cassert>

#include "../../input_iterator.h"

template <class S, class It>
void
test(S s, It first, It last, S expected)
{
    s.append(first, last);
    assert(s.__invariants());
    assert(s == expected);
}

int main()
{
    typedef std::string S;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test(S(), s, s, S());
    test(S(), s, s+1, S("A"));
    test(S(), s, s+10, S("ABCDEFGHIJ"));
    test(S(), s, s+52, S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), s, s, S("12345"));
    test(S("12345"), s, s+1, S("12345A"));
    test(S("12345"), s, s+10, S("12345ABCDEFGHIJ"));
    test(S("12345"), s, s+52, S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), s, s, S("1234567890"));
    test(S("1234567890"), s, s+1, S("1234567890A"));
    test(S("1234567890"), s, s+10, S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), s, s+52, S("1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345678901234567890"), s, s, S("12345678901234567890"));
    test(S("12345678901234567890"), s, s+1, S("12345678901234567890""A"));
    test(S("12345678901234567890"), s, s+10, S("12345678901234567890""ABCDEFGHIJ"));
    test(S("12345678901234567890"), s, s+52,
         S("12345678901234567890""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S(), input_iterator<const char*>(s), input_iterator<const char*>(s), S());
    test(S(), input_iterator<const char*>(s), input_iterator<const char*>(s+1), S("A"));
    test(S(), input_iterator<const char*>(s), input_iterator<const char*>(s+10),
         S("ABCDEFGHIJ"));
    test(S(), input_iterator<const char*>(s), input_iterator<const char*>(s+52),
         S("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345"), input_iterator<const char*>(s), input_iterator<const char*>(s),
         S("12345"));
    test(S("12345"), input_iterator<const char*>(s), input_iterator<const char*>(s+1),
         S("12345A"));
    test(S("12345"), input_iterator<const char*>(s), input_iterator<const char*>(s+10),
         S("12345ABCDEFGHIJ"));
    test(S("12345"), input_iterator<const char*>(s), input_iterator<const char*>(s+52),
         S("12345ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("1234567890"), input_iterator<const char*>(s), input_iterator<const char*>(s),
         S("1234567890"));
    test(S("1234567890"), input_iterator<const char*>(s), input_iterator<const char*>(s+1),
         S("1234567890A"));
    test(S("1234567890"), input_iterator<const char*>(s), input_iterator<const char*>(s+10),
         S("1234567890ABCDEFGHIJ"));
    test(S("1234567890"), input_iterator<const char*>(s), input_iterator<const char*>(s+52),
         S("1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));

    test(S("12345678901234567890"), input_iterator<const char*>(s), input_iterator<const char*>(s),
         S("12345678901234567890"));
    test(S("12345678901234567890"), input_iterator<const char*>(s), input_iterator<const char*>(s+1),
         S("12345678901234567890""A"));
    test(S("12345678901234567890"), input_iterator<const char*>(s), input_iterator<const char*>(s+10),
         S("12345678901234567890""ABCDEFGHIJ"));
    test(S("12345678901234567890"), input_iterator<const char*>(s), input_iterator<const char*>(s+52),
         S("12345678901234567890""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"));
}
