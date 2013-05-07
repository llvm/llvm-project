//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char32_t>

// static constexpr bool lt(char_type c1, char_type c2);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    char32_t c = U'\0';
    assert(!std::char_traits<char32_t>::lt(U'a', U'a'));
    assert( std::char_traits<char32_t>::lt(U'A', U'a'));
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
