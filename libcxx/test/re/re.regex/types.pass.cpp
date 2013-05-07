//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>>
// class basic_regex
// {
// public:
//     // types:
//     typedef charT                               value_type;
//     typedef regex_constants::syntax_option_type flag_type;
//     typedef typename traits::locale_type        locale_type;

#include <regex>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::basic_regex<char>::value_type, char>::value), "");
    static_assert((std::is_same<std::basic_regex<char>::flag_type,
                                std::regex_constants::syntax_option_type>::value), "");
    static_assert((std::is_same<std::basic_regex<char>::locale_type, std::locale>::value), "");

    static_assert((std::is_same<std::basic_regex<wchar_t>::value_type, wchar_t>::value), "");
    static_assert((std::is_same<std::basic_regex<wchar_t>::flag_type,
                                std::regex_constants::syntax_option_type>::value), "");
    static_assert((std::is_same<std::basic_regex<wchar_t>::locale_type, std::locale>::value), "");
}
