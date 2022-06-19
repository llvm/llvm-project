//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char>

// static char_type* copy(char_type* s1, const char_type* s2, size_t n);

#include <string>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test()
{
    char s1[] = {1, 2, 3};
    char s2[3] = {0};
    assert(std::char_traits<char>::copy(s2, s1, 3) == s2);
    assert(s2[0] == char(1));
    assert(s2[1] == char(2));
    assert(s2[2] == char(3));
    assert(std::char_traits<char>::copy(NULL, s1, 0) == NULL);
    assert(std::char_traits<char>::copy(s1, NULL, 0) == s1);

  return true;
}

int main(int, char**)
{
    test();

#if TEST_STD_VER > 17
    static_assert(test());
#endif

  return 0;
}
