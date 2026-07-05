//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class _CharT>
// class messages
//     : public locale::facet,
//       public messages_base
// {
// public:
//     typedef _CharT               char_type;
//     typedef basic_string<_CharT> string_type;

#include <locale>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_base_of<std::locale::facet, std::messages<char> >::value), "");
    static_assert((std::is_base_of<std::messages_base, std::messages<char> >::value), "");
    static_assert((std::is_same<std::messages<char>::char_type, char>::value), "");
    static_assert((std::is_same<std::messages<char>::string_type, std::string>::value), "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    static_assert((std::is_base_of<std::locale::facet, std::messages<wchar_t> >::value), "");
    static_assert((std::is_base_of<std::messages_base, std::messages<wchar_t> >::value), "");
    static_assert((std::is_same<std::messages<wchar_t>::char_type, wchar_t>::value), "");
    static_assert((std::is_same<std::messages<wchar_t>::string_type, std::wstring>::value), "");
#endif

  return 0;
}
