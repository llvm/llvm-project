//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char>

// static size_t length(const char_type* s);

#include <string>
#include <cassert>

int main()
{
    assert(std::char_traits<char>::length("") == 0);
    assert(std::char_traits<char>::length("a") == 1);
    assert(std::char_traits<char>::length("aa") == 2);
    assert(std::char_traits<char>::length("aaa") == 3);
    assert(std::char_traits<char>::length("aaaa") == 4);
}
