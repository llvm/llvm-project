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

// static const char_type* find(const char_type* s, size_t n, const char_type& a);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    char32_t s1[] = {1, 2, 3};
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(1)) == s1);
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(2)) == s1+1);
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(3)) == s1+2);
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(4)) == 0);
    assert(std::char_traits<char32_t>::find(s1, 3, char32_t(0)) == 0);
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
