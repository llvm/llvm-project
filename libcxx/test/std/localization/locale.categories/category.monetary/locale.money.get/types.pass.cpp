//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class CharT, class InputIterator = istreambuf_iterator<CharT> >
// class money_get
//     : public locale::facet
// {
// public:
//     typedef CharT                   char_type;
//     typedef InputIterator           iter_type;
//     typedef basic_string<char_type> string_type;

#include <locale>
#include <type_traits>

int main()
{
    static_assert((std::is_base_of<std::locale::facet, std::money_get<char> >::value), "");
    static_assert((std::is_base_of<std::locale::facet, std::money_get<wchar_t> >::value), "");
    static_assert((std::is_same<std::money_get<char>::char_type, char>::value), "");
    static_assert((std::is_same<std::money_get<wchar_t>::char_type, wchar_t>::value), "");
    static_assert((std::is_same<std::money_get<char>::iter_type, std::istreambuf_iterator<char> >::value), "");
    static_assert((std::is_same<std::money_get<wchar_t>::iter_type, std::istreambuf_iterator<wchar_t> >::value), "");
    static_assert((std::is_same<std::money_get<char>::string_type, std::string>::value), "");
    static_assert((std::is_same<std::money_get<wchar_t>::string_type, std::wstring>::value), "");
}
