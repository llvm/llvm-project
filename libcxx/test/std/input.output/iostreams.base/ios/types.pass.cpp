//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits = char_traits<charT> >
// class basic_ios : public ios_base
// {
// public:
//     typedef charT char_type;
//     typedef typename traits::int_type int_type;
//     typedef typename traits::pos_type pos_type;
//     typedef typename traits::off_type off_type;
//     typedef traits traits_type;

#include <ios>
#include <type_traits>

int main(int, char**)
{
    static_assert((std::is_base_of<std::ios_base, std::basic_ios<char> >::value), "");
    static_assert((std::is_same<std::basic_ios<char>::char_type, char>::value), "");
    static_assert((std::is_same<std::basic_ios<char>::traits_type, std::char_traits<char> >::value), "");
    static_assert((std::is_same<std::basic_ios<char>::int_type, std::char_traits<char>::int_type>::value), "");
    static_assert((std::is_same<std::basic_ios<char>::pos_type, std::char_traits<char>::pos_type>::value), "");
    static_assert((std::is_same<std::basic_ios<char>::off_type, std::char_traits<char>::off_type>::value), "");

  return 0;
}
