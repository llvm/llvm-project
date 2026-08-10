//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string<charT,traits,Allocator>&
//   operator=(const basic_string<charT,traits,Allocator>& str); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s1, const S& s2) {
  s1 = s2;
  LIBCPP_ASSERT(s1.__invariants());
  assert(s1 == s2);
  assert(s1.capacity() >= s1.size());
  LIBCPP_ASSERT(is_string_asan_correct(s1));
  LIBCPP_ASSERT(is_string_asan_correct(s2));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(), S());
  test(S("1"), S());
  test(S(), S("1"));
  test(S("1"), S("2"));
  test(S("1"), S("2"));

  test(S(), S("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"));
  test(S("123456789"), S("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"));
  test(S("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"), S("123456789"));
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890"),
       S("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"));
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890"
         "1234567890123456789012345678901234567890123456789012345678901234567890"),
       S("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char>>>();
#endif

#if TEST_STD_VER >= 11
  { // LWG 2946
    std::string s;
    s = {"abc", 1};
    assert(s.size() == 1);
    assert(s == "a");
  }
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
