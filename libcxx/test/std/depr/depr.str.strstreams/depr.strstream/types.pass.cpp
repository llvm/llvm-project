//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstream
//     : public basic_iostream<char>
// {
// public:
//     // Types
//     typedef char                        char_type;
//     typedef char_traits<char>::int_type int_type;
//     typedef char_traits<char>::pos_type pos_type;
//     typedef char_traits<char>::off_type off_type;

#include <strstream>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_base_of<std::iostream, std::strstream>::value), "");
    static_assert((std::is_same<std::strstream::char_type, char>::value), "");
    static_assert((std::is_same<std::strstream::int_type, std::char_traits<char>::int_type>::value), "");
    static_assert((std::is_same<std::strstream::pos_type, std::char_traits<char>::pos_type>::value), "");
    static_assert((std::is_same<std::strstream::off_type, std::char_traits<char>::off_type>::value), "");

  return 0;
}
