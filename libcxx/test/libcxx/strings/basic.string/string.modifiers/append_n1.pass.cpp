//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <string>

#include "test_macros.h"

struct n1_char_traits : std::char_traits<char> {
  static TEST_CONSTEXPR_CXX20 char* copy(char* __dest, const char* __source, std::size_t __count) {
    assert(__count != 1);
    return std::char_traits<char>::copy(__dest, __source, __count);
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  std::basic_string<char, n1_char_traits> __s;
  __s.append("x", 1);
  assert(__s.size() == 1);
  assert(__s[0] == 'x');
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
