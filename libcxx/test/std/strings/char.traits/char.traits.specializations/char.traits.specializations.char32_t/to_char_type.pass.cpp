//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char32_t>

// static constexpr char_type to_char_type(int_type c);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
#if TEST_STD_VER >= 11
  assert(std::char_traits<char32_t>::to_char_type(U'a') == U'a');
  assert(std::char_traits<char32_t>::to_char_type(U'A') == U'A');
#endif
  assert(std::char_traits<char32_t>::to_char_type(0) == 0);

  return 0;
}
