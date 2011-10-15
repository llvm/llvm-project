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

// static constexpr bool lt(char_type c1, char_type c2);

#include <string>
#include <cassert>

int main()
{
    char c = '\0';
    assert(!std::char_traits<char>::lt('a', 'a'));
    assert( std::char_traits<char>::lt('A', 'a'));
    assert(!std::char_traits<char>::lt('A' + 127, 'a'));
    assert(!std::char_traits<char>::lt('A' - 127, 'a'));
    assert( std::char_traits<char>::lt('A', 'a' + 127));
    assert( std::char_traits<char>::lt('A', 'a' - 127));
}
