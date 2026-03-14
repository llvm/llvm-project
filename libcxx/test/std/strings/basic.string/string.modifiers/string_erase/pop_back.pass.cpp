//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void pop_back(); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s, S expected) {
  s.pop_back();
  LIBCPP_ASSERT(s.__invariants());
  assert(s[s.size()] == typename S::value_type());
  assert(s == expected);
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S("abcde"), S("abcd"));
  test(S("abcdefghij"), S("abcdefghi"));
  test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrs"));
  test(S("abcdefghijklmnopqrstabcdefghijklmnopqrstabcdefghijklmnopqrstabcdefghijklmnopqrst"),
       S("abcdefghijklmnopqrstabcdefghijklmnopqrstabcdefghijklmnopqrstabcdefghijklmnopqrs"));
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
