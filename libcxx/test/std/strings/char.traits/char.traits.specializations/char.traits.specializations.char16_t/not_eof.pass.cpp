//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char16_t>

// static constexpr int_type not_eof(int_type c);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_STD_VER >= 11
    assert(std::char_traits<char16_t>::not_eof(u'a') == u'a');
    assert(std::char_traits<char16_t>::not_eof(u'A') == u'A');
#endif
    assert(std::char_traits<char16_t>::not_eof(0) == 0);
    assert(std::char_traits<char16_t>::not_eof(std::char_traits<char16_t>::eof()) !=
           std::char_traits<char16_t>::eof());

  return 0;
}
