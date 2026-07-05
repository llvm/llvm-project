//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class _CharT, bool _International = false>
// class moneypunct
//     : public locale::facet,
//       public money_base
// {
// public:
//     typedef _CharT                  char_type;
//     typedef basic_string<char_type> string_type;
//     static const bool intl = International;

#include <locale>
#include <type_traits>

#include "test_macros.h"

template <class T>
void test(const T &) {}

int main(int, char**)
{
    static_assert((std::is_base_of<std::locale::facet, std::moneypunct<char> >::value), "");
    static_assert((std::is_base_of<std::money_base, std::moneypunct<char> >::value), "");
    static_assert((std::is_same<std::moneypunct<char>::char_type, char>::value), "");
    static_assert((std::is_same<std::moneypunct<char>::string_type, std::string>::value), "");
    test(std::moneypunct<char, false>::intl);
    test(std::moneypunct<char, true>::intl);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    static_assert((std::is_base_of<std::locale::facet, std::moneypunct<wchar_t> >::value), "");
    static_assert((std::is_base_of<std::money_base, std::moneypunct<wchar_t> >::value), "");
    static_assert((std::is_same<std::moneypunct<wchar_t>::char_type, wchar_t>::value), "");
    static_assert((std::is_same<std::moneypunct<wchar_t>::string_type, std::wstring>::value), "");
    test(std::moneypunct<wchar_t, false>::intl);
    test(std::moneypunct<wchar_t, true>::intl);
#endif

  return 0;
}
