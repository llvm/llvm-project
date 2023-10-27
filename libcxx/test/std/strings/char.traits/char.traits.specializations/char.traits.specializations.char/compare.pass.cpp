//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char>

// static int compare(const char_type* s1, const char_type* s2, size_t n);
// constexpr in C++17

#include <string>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX17 bool test() {
  assert(std::char_traits<char>::compare("", "", 0) == 0);
  assert(std::char_traits<char>::compare(NULL, NULL, 0) == 0);

  assert(std::char_traits<char>::compare("1", "1", 1) == 0);
  assert(std::char_traits<char>::compare("1", "2", 1) < 0);
  assert(std::char_traits<char>::compare("2", "1", 1) > 0);

  assert(std::char_traits<char>::compare("12", "12", 2) == 0);
  assert(std::char_traits<char>::compare("12", "13", 2) < 0);
  assert(std::char_traits<char>::compare("12", "22", 2) < 0);
  assert(std::char_traits<char>::compare("13", "12", 2) > 0);
  assert(std::char_traits<char>::compare("22", "12", 2) > 0);

  assert(std::char_traits<char>::compare("123", "123", 3) == 0);
  assert(std::char_traits<char>::compare("123", "223", 3) < 0);
  assert(std::char_traits<char>::compare("123", "133", 3) < 0);
  assert(std::char_traits<char>::compare("123", "124", 3) < 0);
  assert(std::char_traits<char>::compare("223", "123", 3) > 0);
  assert(std::char_traits<char>::compare("133", "123", 3) > 0);
  assert(std::char_traits<char>::compare("124", "123", 3) > 0);

  {
    char a[] = {static_cast<char>(-1), 0};
    char b[] = {1, 0};
    assert(std::char_traits<char>::compare(a, b, 1) > 0);
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 17
  static_assert(test());
#endif

  return 0;
}
