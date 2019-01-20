//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>>
// class basic_regex
// {
// public:
//     // types:
//     typedef charT                               value_type;
//     typedef traits                              traits_type;
//     typedef typename traits::string_type        string_type;
//     typedef regex_constants::syntax_option_type flag_type;
//     typedef typename traits::locale_type        locale_type;

#include <regex>
#include <type_traits>
#include "test_macros.h"

int main()
{
    static_assert((std::is_same<std::basic_regex<char>::value_type, char>::value), "");
    static_assert((std::is_same<std::basic_regex<char>::traits_type, std::regex_traits<char> >::value), "");
    static_assert((std::is_same<std::basic_regex<char>::string_type, std::basic_string<char> >::value), "");
    static_assert((std::is_same<std::basic_regex<char>::flag_type,
                                std::regex_constants::syntax_option_type>::value), "");
    static_assert((std::is_same<std::basic_regex<char>::locale_type, std::locale>::value), "");

    static_assert((std::is_same<std::basic_regex<wchar_t>::value_type, wchar_t>::value), "");
    static_assert((std::is_same<std::basic_regex<wchar_t>::traits_type, std::regex_traits<wchar_t> >::value), "");
    static_assert((std::is_same<std::basic_regex<wchar_t>::string_type, std::basic_string<wchar_t> >::value), "");
    static_assert((std::is_same<std::basic_regex<wchar_t>::flag_type,
                                std::regex_constants::syntax_option_type>::value), "");
    static_assert((std::is_same<std::basic_regex<wchar_t>::locale_type, std::locale>::value), "");
}
