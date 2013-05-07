//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// Test for the existence of:

// basic_string typedef names
// typedef basic_string<char> string;
// typedef basic_string<char16_t> u16string;
// typedef basic_string<char32_t> u32string;
// typedef basic_string<wchar_t> wstring;

#include <string>

int main()
{
    typedef std::string test1;
    typedef std::wstring test2;
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    typedef std::u16string test3;
    typedef std::u32string test4;
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
