//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char16_t>

// static constexpr bool lt(char_type c1, char_type c2);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#ifndef TEST_HAS_NO_UNICODE_CHARS
#if TEST_STD_VER >= 11
    assert(!std::char_traits<char16_t>::lt(u'a', u'a'));
    assert( std::char_traits<char16_t>::lt(u'A', u'a'));
#endif
#endif // TEST_HAS_NO_UNICODE_CHARS

  return 0;
}
