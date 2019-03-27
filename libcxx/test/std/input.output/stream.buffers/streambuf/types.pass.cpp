//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf
// {
// public:
//     // types:
//     typedef charT char_type;
//     typedef traits traits_type;
//     typedef typename traits_type::int_type int_type;
//     typedef typename traits_type::pos_type pos_type;
//     typedef typename traits_type::off_type off_type;

#include <streambuf>
#include <type_traits>

int main(int, char**)
{
    static_assert((std::is_same<std::streambuf::char_type, char>::value), "");
    static_assert((std::is_same<std::streambuf::traits_type, std::char_traits<char> >::value), "");
    static_assert((std::is_same<std::streambuf::int_type, std::char_traits<char>::int_type>::value), "");
    static_assert((std::is_same<std::streambuf::pos_type, std::char_traits<char>::pos_type>::value), "");
    static_assert((std::is_same<std::streambuf::off_type, std::char_traits<char>::off_type>::value), "");

    static_assert((std::is_same<std::wstreambuf::char_type, wchar_t>::value), "");
    static_assert((std::is_same<std::wstreambuf::traits_type, std::char_traits<wchar_t> >::value), "");
    static_assert((std::is_same<std::wstreambuf::int_type, std::char_traits<wchar_t>::int_type>::value), "");
    static_assert((std::is_same<std::wstreambuf::pos_type, std::char_traits<wchar_t>::pos_type>::value), "");
    static_assert((std::is_same<std::wstreambuf::off_type, std::char_traits<wchar_t>::off_type>::value), "");

  return 0;
}
