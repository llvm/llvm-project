//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ifstream
//     : public basic_istream<charT,traits>
// {
// public:
//     typedef charT                          char_type;
//     typedef traits                         traits_type;
//     typedef typename traits_type::int_type int_type;
//     typedef typename traits_type::pos_type pos_type;
//     typedef typename traits_type::off_type off_type;

#include <fstream>
#include <type_traits>

#include "test_macros.h"
#include "../native_handle_test_helpers.h"

int main(int, char**)
{
    static_assert((std::is_base_of<std::basic_istream<char>, std::basic_ifstream<char> >::value), "");
    static_assert((std::is_same<std::basic_ifstream<char>::char_type, char>::value), "");
    static_assert((std::is_same<std::basic_ifstream<char>::traits_type, std::char_traits<char> >::value), "");
    static_assert((std::is_same<std::basic_ifstream<char>::int_type, std::char_traits<char>::int_type>::value), "");
    static_assert((std::is_same<std::basic_ifstream<char>::pos_type, std::char_traits<char>::pos_type>::value), "");
    static_assert((std::is_same<std::basic_ifstream<char>::off_type, std::char_traits<char>::off_type>::value), "");
#if TEST_STD_VER >= 26
    test_native_handle_type< std::basic_ifstream<char>>();
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test_native_handle_type< std::basic_ifstream<wchar_t>>();
#  endif
#endif

    return 0;
}
