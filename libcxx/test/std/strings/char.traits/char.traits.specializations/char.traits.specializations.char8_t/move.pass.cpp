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

// static char_type* move(char_type* s1, const char_type* s2, size_t n);

#include <string>
#include <cassert>

int main(int, char**)
{
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    char8_t s1[] = {1, 2, 3};
    assert(std::char_traits<char8_t>::move(s1, s1+1, 2) == s1);
    assert(s1[0] == char8_t(2));
    assert(s1[1] == char8_t(3));
    assert(s1[2] == char8_t(3));
    s1[2] = char8_t(0);
    assert(std::char_traits<char8_t>::move(s1+1, s1, 2) == s1+1);
    assert(s1[0] == char8_t(2));
    assert(s1[1] == char8_t(2));
    assert(s1[2] == char8_t(3));
    assert(std::char_traits<char8_t>::move(NULL, s1, 0) == NULL);
    assert(std::char_traits<char8_t>::move(s1, NULL, 0) == s1);
#endif

  return 0;
}
