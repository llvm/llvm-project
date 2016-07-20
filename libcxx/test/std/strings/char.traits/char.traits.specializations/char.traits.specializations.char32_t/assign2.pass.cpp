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

// static void assign(char_type& c1, const char_type& c2);

#include <string>
#include <cassert>

#include "test_macros.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
#if TEST_STD_VER >= 11
    char32_t c = U'\0';
    std::char_traits<char32_t>::assign(c, U'a');
    assert(c == U'a');
#endif
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
