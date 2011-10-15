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

// static constexpr bool eq_int_type(int_type c1, int_type c2);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    assert( std::char_traits<char32_t>::eq_int_type(U'a', U'a'));
    assert(!std::char_traits<char32_t>::eq_int_type(U'a', U'A'));
    assert(!std::char_traits<char32_t>::eq_int_type(std::char_traits<char32_t>::eof(), U'A'));
    assert( std::char_traits<char32_t>::eq_int_type(std::char_traits<char32_t>::eof(),
                                                    std::char_traits<char32_t>::eof()));
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
