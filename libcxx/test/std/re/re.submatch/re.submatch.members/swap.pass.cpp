//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class BidirectionalIterator> class sub_match;

// void swap(sub_match& s) noexcept(see below);

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**) {
  {
    using CharT      = char;
    using SM         = std::sub_match<const CharT*>;
    const CharT s1[] = {'1', '2', '3', 0};
    SM sm1;
    sm1.first   = s1;
    sm1.second  = s1 + 3;
    sm1.matched = true;

    SM sm2;
    const CharT s2[] = {'c', 'a', 't', 0};
    sm2.first        = s2;
    sm2.second       = s2 + 3;
    sm2.matched      = false;

    sm1.swap(sm2);

    assert(sm1.first == s2);
    assert(sm1.second == s2 + 3);
    assert(!sm1.matched);

    assert(sm2.first == s1);
    assert(sm2.second == s1 + 3);
    assert(sm2.matched);

    ASSERT_NOEXCEPT(sm1.swap(sm2));
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    using CharT      = wchar_t;
    using SM         = std::sub_match<const CharT*>;
    const CharT s1[] = {L'1', L'2', L'3', 0};
    SM sm1;
    sm1.first   = s1;
    sm1.second  = s1 + 3;
    sm1.matched = true;

    SM sm2;
    const CharT s2[] = {L'c', L'a', L't', 0};
    sm2.first        = s2;
    sm2.second       = s2 + 3;
    sm2.matched      = false;

    sm1.swap(sm2);

    assert(sm1.first == s2);
    assert(sm1.second == s2 + 3);
    assert(!sm1.matched);

    assert(sm2.first == s1);
    assert(sm2.second == s1 + 3);
    assert(sm2.matched);

    ASSERT_NOEXCEPT(sm1.swap(sm2));
  }
#endif
}
