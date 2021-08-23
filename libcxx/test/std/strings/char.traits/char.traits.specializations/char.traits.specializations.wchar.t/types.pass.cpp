//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<wchar_t>

// typedef wchar_t   char_type;
// typedef int       int_type;
// typedef streamoff off_type;
// typedef streampos pos_type;
// typedef mbstate_t state_type;

// UNSUPPORTED: libcpp-has-no-wide-characters

#include <string>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<std::char_traits<wchar_t>::char_type, wchar_t>::value), "");
    static_assert((std::is_same<std::char_traits<wchar_t>::int_type, std::wint_t>::value), "");
    static_assert((std::is_same<std::char_traits<wchar_t>::off_type, std::streamoff>::value), "");
    static_assert((std::is_same<std::char_traits<wchar_t>::pos_type, std::wstreampos>::value), "");
    static_assert((std::is_same<std::char_traits<wchar_t>::state_type, std::mbstate_t>::value), "");

  return 0;
}
