//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char>

// static constexpr bool lt(char_type c1, char_type c2);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  assert(std::char_traits<char>::lt('\0', 'A') == ('\0' < 'A'));
  assert(std::char_traits<char>::lt('A', '\0') == ('A' < '\0'));

  assert(std::char_traits<char>::lt('a', 'a') == ('a' < 'a'));
  assert(std::char_traits<char>::lt('A', 'a') == ('A' < 'a'));
  assert(std::char_traits<char>::lt('a', 'A') == ('a' < 'A'));

  assert(std::char_traits<char>::lt('a', 'z') == ('a' < 'z'));
  assert(std::char_traits<char>::lt('A', 'Z') == ('A' < 'Z'));

  assert(std::char_traits<char>::lt(' ', 'A') == (' ' < 'A'));
  assert(std::char_traits<char>::lt('A', '~') == ('A' < '~'));

  return 0;
}
