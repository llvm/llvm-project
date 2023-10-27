//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char16_t>

// static constexpr int_type to_int_type(char_type c);

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
#if TEST_STD_VER >= 11
  assert(std::char_traits<char16_t>::to_int_type(u'a') == u'a');
  assert(std::char_traits<char16_t>::to_int_type(u'A') == u'A');
#endif
  assert(std::char_traits<char16_t>::to_int_type(0) == 0);

  return 0;
}
