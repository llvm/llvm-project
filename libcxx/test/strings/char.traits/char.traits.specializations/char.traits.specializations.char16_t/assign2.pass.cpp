//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char16_t>

// static void assign(char_type& c1, const char_type& c2);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    char16_t c = u'\0';
    std::char_traits<char16_t>::assign(c, u'a');
    assert(c == u'a');
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
