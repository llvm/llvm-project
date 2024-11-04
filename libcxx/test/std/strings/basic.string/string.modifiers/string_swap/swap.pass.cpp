//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void swap(basic_string& s); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s1, S s2) {
  S s1_ = s1;
  S s2_ = s2;
  s1.swap(s2);
  LIBCPP_ASSERT(s1.__invariants());
  LIBCPP_ASSERT(s2.__invariants());
  assert(s1 == s2_);
  assert(s2 == s1_);
  LIBCPP_ASSERT(is_string_asan_correct(s1));
  LIBCPP_ASSERT(is_string_asan_correct(s2));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(""), S(""));
  test(S(""), S("12345"));
  test(S(""), S("1234567890"));
  test(S(""), S("12345678901234567890"));
  test(S("abcde"), S(""));
  test(S("abcde"), S("12345"));
  test(S("abcde"), S("1234567890"));
  test(S("abcde"), S("12345678901234567890"));
  test(S("abcdefghij"), S(""));
  test(S("abcdefghij"), S("12345"));
  test(S("abcdefghij"), S("1234567890"));
  test(S("abcdefghij"), S("12345678901234567890"));
  test(S("abcdefghijklmnopqrst"), S(""));
  test(S("abcdefghijklmnopqrst"), S("12345"));
  test(S("abcdefghijklmnopqrst"), S("1234567890"));
  test(S("abcdefghijklmnopqrst"), S("12345678901234567890"));
  test(S("abcdefghijklmnopqrstabcdefghijklmnopqrst"), S(""));
  test(S("abcdefghijklmnopqrstabcdefghijklmnopqrst"), S("12345"));
  test(S("abcdefghijklmnopqrstabcdefghijklmnopqrst"), S("1234567890"));
  test(S("abcdefghijklmnopqrstabcdefghijklmnopqrst"), S("12345678901234567890"));
  test(S("abcdefghijklmnopqrstabcdefghijklmnopqrst"), S("12345678901234567890abcdefghijklmnopqrst"));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char>>>();
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
