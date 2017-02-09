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
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::string, std::basic_string<char> >::value), "");
    static_assert((std::is_same<std::wstring, std::basic_string<wchar_t> >::value), "");
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    static_assert((std::is_same<std::u16string, std::basic_string<char16_t> >::value), "");
    static_assert((std::is_same<std::u32string, std::basic_string<char32_t> >::value), "");
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
