//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<wchar_t>

// static constexpr bool lt(char_type c1, char_type c2);

// UNSUPPORTED: no-wide-characters

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  assert(std::char_traits<wchar_t>::lt(L'\0', L'A') == (L'\0' < L'A'));
  assert(std::char_traits<wchar_t>::lt(L'A', L'\0') == (L'A' < L'\0'));

  assert(std::char_traits<wchar_t>::lt(L'a', L'a') == (L'a' < L'a'));
  assert(std::char_traits<wchar_t>::lt(L'A', L'a') == (L'A' < L'a'));
  assert(std::char_traits<wchar_t>::lt(L'a', L'A') == (L'a' < L'A'));

  assert(std::char_traits<wchar_t>::lt(L'a', L'z') == (L'a' < L'z'));
  assert(std::char_traits<wchar_t>::lt(L'A', L'Z') == (L'A' < L'Z'));

  assert(std::char_traits<wchar_t>::lt(L' ', L'A') == (L' ' < L'A'));
  assert(std::char_traits<wchar_t>::lt(L'A', L'~') == (L'A' < L'~'));

  return 0;
}
