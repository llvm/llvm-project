//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char16_t>

// typedef char16_t       char_type;
// typedef uint_least16_t int_type;
// typedef streamoff      off_type;
// typedef u16streampos   pos_type;
// typedef mbstate_t      state_type;

#include <string>
#include <type_traits>
#include <cstdint>

int main()
{
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    static_assert((std::is_same<std::char_traits<char16_t>::char_type, char16_t>::value), "");
    static_assert((std::is_same<std::char_traits<char16_t>::int_type, std::uint_least16_t>::value), "");
    static_assert((std::is_same<std::char_traits<char16_t>::off_type, std::streamoff>::value), "");
    static_assert((std::is_same<std::char_traits<char16_t>::pos_type, std::u16streampos>::value), "");
    static_assert((std::is_same<std::char_traits<char16_t>::state_type, std::mbstate_t>::value), "");
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
