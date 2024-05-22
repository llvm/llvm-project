//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string>

// template<> struct char_traits<char8_t>

// static constexpr bool eq(char_type c1, char_type c2);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
#ifndef TEST_HAS_NO_CHAR8_T
  assert(std::char_traits<char8_t>::eq(u8'a', u8'a'));
  assert(!std::char_traits<char8_t>::eq(u8'a', u8'A'));
#endif

  return 0;
}
