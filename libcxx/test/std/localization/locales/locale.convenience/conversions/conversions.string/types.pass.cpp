//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template<class Codecvt, class Elem = wchar_t,
//          class Wide_alloc = allocator<Elem>,
//          class Byte_alloc = allocator<char>>
// class wstring_convert
// {
// public:
//     typedef basic_string<char, char_traits<char>, Byte_alloc> byte_string;
//     typedef basic_string<Elem, char_traits<Elem>, Wide_alloc> wide_string;
//     typedef typename Codecvt::state_type                      state_type;
//     typedef typename wide_string::traits_type::int_type       int_type;

#include <locale>
#include <codecvt>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::wstring_convert<std::codecvt_utf8<wchar_t> > myconv;
        static_assert((std::is_same<myconv::byte_string, std::string>::value), "");
        static_assert((std::is_same<myconv::wide_string, std::wstring>::value), "");
        static_assert((std::is_same<myconv::state_type, std::mbstate_t>::value), "");
        static_assert((std::is_same<myconv::int_type, std::char_traits<wchar_t>::int_type>::value), "");
    }

  return 0;
}
