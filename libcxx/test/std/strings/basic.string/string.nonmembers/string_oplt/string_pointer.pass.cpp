//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   bool operator<(const basic_string<charT,traits,Allocator>& lhs, const charT* rhs); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(const S& lhs, const typename S::value_type* rhs, bool x) {
  assert((lhs < rhs) == x);
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(""), "", false);
  test(S(""), "abcde", true);
  test(S(""), "abcdefghij", true);
  test(S(""), "abcdefghijklmnopqrst", true);
  test(S("abcde"), "", false);
  test(S("abcde"), "abcde", false);
  test(S("abcde"), "abcdefghij", true);
  test(S("abcde"), "abcdefghijklmnopqrst", true);
  test(S("abcdefghij"), "", false);
  test(S("abcdefghij"), "abcde", false);
  test(S("abcdefghij"), "abcdefghij", false);
  test(S("abcdefghij"), "abcdefghijklmnopqrst", true);
  test(S("abcdefghijklmnopqrst"), "", false);
  test(S("abcdefghijklmnopqrst"), "abcde", false);
  test(S("abcdefghijklmnopqrst"), "abcdefghij", false);
  test(S("abcdefghijklmnopqrst"), "abcdefghijklmnopqrst", false);
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
