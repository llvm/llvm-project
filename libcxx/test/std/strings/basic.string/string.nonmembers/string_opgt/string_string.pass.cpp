//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   bool operator>(const basic_string<charT,traits,Allocator>& lhs,
//                  const basic_string<charT,traits,Allocator>& rhs); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(const S& lhs, const S& rhs, bool x) {
  assert((lhs > rhs) == x);
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(""), S(""), false);
  test(S(""), S("abcde"), false);
  test(S(""), S("abcdefghij"), false);
  test(S(""), S("abcdefghijklmnopqrst"), false);
  test(S("abcde"), S(""), true);
  test(S("abcde"), S("abcde"), false);
  test(S("abcde"), S("abcdefghij"), false);
  test(S("abcde"), S("abcdefghijklmnopqrst"), false);
  test(S("abcdefghij"), S(""), true);
  test(S("abcdefghij"), S("abcde"), true);
  test(S("abcdefghij"), S("abcdefghij"), false);
  test(S("abcdefghij"), S("abcdefghijklmnopqrst"), false);
  test(S("abcdefghijklmnopqrst"), S(""), true);
  test(S("abcdefghijklmnopqrst"), S("abcde"), true);
  test(S("abcdefghijklmnopqrst"), S("abcdefghij"), true);
  test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrst"), false);
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
