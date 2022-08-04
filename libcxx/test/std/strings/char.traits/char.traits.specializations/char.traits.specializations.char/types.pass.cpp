//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char>

// typedef char      char_type;
// typedef int       int_type;
// typedef streamoff off_type;
// typedef streampos pos_type;
// typedef mbstate_t state_type;
// using comparison_category = strong_ordering;

#include <string>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<std::char_traits<char>::char_type, char>::value), "");
    static_assert((std::is_same<std::char_traits<char>::int_type, int>::value), "");
    static_assert((std::is_same<std::char_traits<char>::off_type, std::streamoff>::value), "");
    static_assert((std::is_same<std::char_traits<char>::pos_type, std::streampos>::value), "");
    static_assert((std::is_same<std::char_traits<char>::state_type, std::mbstate_t>::value), "");
#if TEST_STD_VER > 17
    static_assert(std::is_same_v<std::char_traits<char>::comparison_category, std::strong_ordering>);
#endif

  return 0;
}
