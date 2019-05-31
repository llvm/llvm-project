//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <string>

// template<> struct char_traits<char8_t>

// static char_type* copy(char_type* s1, const char_type* s2, size_t n);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    char8_t s1[] = {1, 2, 3};
    char8_t s2[3] = {0};
    assert(std::char_traits<char8_t>::copy(s2, s1, 3) == s2);
    assert(s2[0] == char8_t(1));
    assert(s2[1] == char8_t(2));
    assert(s2[2] == char8_t(3));
    assert(std::char_traits<char8_t>::copy(NULL, s1, 0) == NULL);
    assert(std::char_traits<char8_t>::copy(s1, NULL, 0) == s1);
#endif

  return 0;
}
