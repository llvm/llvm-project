//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// size_type rfind(const basic_string& str, size_type pos = npos) const; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(const S& s, const S& str, typename S::size_type pos, typename S::size_type x) {
  LIBCPP_ASSERT_NOEXCEPT(s.rfind(str, pos));
  assert(s.rfind(str, pos) == x);
  if (x != S::npos)
    assert(x <= pos && x + str.size() <= s.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void test(const S& s, const S& str, typename S::size_type x) {
  LIBCPP_ASSERT_NOEXCEPT(s.rfind(str));
  assert(s.rfind(str) == x);
  if (x != S::npos)
    assert(0 <= x && x + str.size() <= s.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  {
    test(S(""), S(""), 0, 0);
    test(S(""), S("abcde"), 0, S::npos);
    test(S(""), S("abcdeabcde"), 0, S::npos);
    test(S(""), S("abcdeabcdeabcdeabcde"), 0, S::npos);
    test(S(""), S(""), 1, 0);
    test(S(""), S("abcde"), 1, S::npos);
    test(S(""), S("abcdeabcde"), 1, S::npos);
    test(S(""), S("abcdeabcdeabcdeabcde"), 1, S::npos);
    test(S("abcde"), S(""), 0, 0);
    test(S("abcde"), S("abcde"), 0, 0);
    test(S("abcde"), S("abcdeabcde"), 0, S::npos);
    test(S("abcde"), S("abcdeabcdeabcdeabcde"), 0, S::npos);
    test(S("abcde"), S(""), 1, 1);
    test(S("abcde"), S("abcde"), 1, 0);
    test(S("abcde"), S("abcdeabcde"), 1, S::npos);
    test(S("abcde"), S("abcdeabcdeabcdeabcde"), 1, S::npos);
    test(S("abcde"), S(""), 2, 2);
    test(S("abcde"), S("abcde"), 2, 0);
    test(S("abcde"), S("abcdeabcde"), 2, S::npos);
    test(S("abcde"), S("abcdeabcdeabcdeabcde"), 2, S::npos);
    test(S("abcde"), S(""), 4, 4);
    test(S("abcde"), S("abcde"), 4, 0);
    test(S("abcde"), S("abcdeabcde"), 4, S::npos);
    test(S("abcde"), S("abcdeabcdeabcdeabcde"), 4, S::npos);
    test(S("abcde"), S(""), 5, 5);
    test(S("abcde"), S("abcde"), 5, 0);
    test(S("abcde"), S("abcdeabcde"), 5, S::npos);
    test(S("abcde"), S("abcdeabcdeabcdeabcde"), 5, S::npos);
    test(S("abcde"), S(""), 6, 5);
    test(S("abcde"), S("abcde"), 6, 0);
    test(S("abcde"), S("abcdeabcde"), 6, S::npos);
    test(S("abcde"), S("abcdeabcdeabcdeabcde"), 6, S::npos);
    test(S("abcdeabcde"), S(""), 0, 0);
    test(S("abcdeabcde"), S("abcde"), 0, 0);
    test(S("abcdeabcde"), S("abcdeabcde"), 0, 0);
    test(S("abcdeabcde"), S("abcdeabcdeabcdeabcde"), 0, S::npos);
    test(S("abcdeabcde"), S(""), 1, 1);
    test(S("abcdeabcde"), S("abcde"), 1, 0);
    test(S("abcdeabcde"), S("abcdeabcde"), 1, 0);
    test(S("abcdeabcde"), S("abcdeabcdeabcdeabcde"), 1, S::npos);
    test(S("abcdeabcde"), S(""), 5, 5);
    test(S("abcdeabcde"), S("abcde"), 5, 5);
    test(S("abcdeabcde"), S("abcdeabcde"), 5, 0);
    test(S("abcdeabcde"), S("abcdeabcdeabcdeabcde"), 5, S::npos);
    test(S("abcdeabcde"), S(""), 9, 9);
    test(S("abcdeabcde"), S("abcde"), 9, 5);
    test(S("abcdeabcde"), S("abcdeabcde"), 9, 0);
    test(S("abcdeabcde"), S("abcdeabcdeabcdeabcde"), 9, S::npos);
    test(S("abcdeabcde"), S(""), 10, 10);
    test(S("abcdeabcde"), S("abcde"), 10, 5);
    test(S("abcdeabcde"), S("abcdeabcde"), 10, 0);
    test(S("abcdeabcde"), S("abcdeabcdeabcdeabcde"), 10, S::npos);
    test(S("abcdeabcde"), S(""), 11, 10);
    test(S("abcdeabcde"), S("abcde"), 11, 5);
    test(S("abcdeabcde"), S("abcdeabcde"), 11, 0);
    test(S("abcdeabcde"), S("abcdeabcdeabcdeabcde"), 11, S::npos);
    test(S("abcdeabcdeabcdeabcde"), S(""), 0, 0);
    test(S("abcdeabcdeabcdeabcde"), S("abcde"), 0, 0);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcde"), 0, 0);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcdeabcdeabcde"), 0, 0);
    test(S("abcdeabcdeabcdeabcde"), S(""), 1, 1);
    test(S("abcdeabcdeabcdeabcde"), S("abcde"), 1, 0);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcde"), 1, 0);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcdeabcdeabcde"), 1, 0);
    test(S("abcdeabcdeabcdeabcde"), S(""), 10, 10);
    test(S("abcdeabcdeabcdeabcde"), S("abcde"), 10, 10);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcde"), 10, 10);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcdeabcdeabcde"), 10, 0);
    test(S("abcdeabcdeabcdeabcde"), S(""), 19, 19);
    test(S("abcdeabcdeabcdeabcde"), S("abcde"), 19, 15);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcde"), 19, 10);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcdeabcdeabcde"), 19, 0);
    test(S("abcdeabcdeabcdeabcde"), S(""), 20, 20);
    test(S("abcdeabcdeabcdeabcde"), S("abcde"), 20, 15);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcde"), 20, 10);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcdeabcdeabcde"), 20, 0);
    test(S("abcdeabcdeabcdeabcde"), S(""), 21, 20);
    test(S("abcdeabcdeabcdeabcde"), S("abcde"), 21, 15);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcde"), 21, 10);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcdeabcdeabcde"), 21, 0);
  }
  {
    test(S(""), S(""), 0);
    test(S(""), S("abcde"), S::npos);
    test(S(""), S("abcdeabcde"), S::npos);
    test(S(""), S("abcdeabcdeabcdeabcde"), S::npos);
    test(S("abcde"), S(""), 5);
    test(S("abcde"), S("abcde"), 0);
    test(S("abcde"), S("abcdeabcde"), S::npos);
    test(S("abcde"), S("abcdeabcdeabcdeabcde"), S::npos);
    test(S("abcdeabcde"), S(""), 10);
    test(S("abcdeabcde"), S("abcde"), 5);
    test(S("abcdeabcde"), S("abcdeabcde"), 0);
    test(S("abcdeabcde"), S("abcdeabcdeabcdeabcde"), S::npos);
    test(S("abcdeabcdeabcdeabcde"), S(""), 20);
    test(S("abcdeabcdeabcdeabcde"), S("abcde"), 15);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcde"), 10);
    test(S("abcdeabcdeabcdeabcde"), S("abcdeabcdeabcdeabcde"), 0);
  }
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
  { // LWG 2946
    std::string s = " !";
    assert(s.rfind({"abc", 1}) == std::string::npos);
  }
#endif

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif

  return 0;
}
