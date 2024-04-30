//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// size_type find(charT c, size_type pos = 0) const; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(const S& s, typename S::value_type c, typename S::size_type pos, typename S::size_type x) {
  LIBCPP_ASSERT_NOEXCEPT(s.find(c, pos));
  assert(s.find(c, pos) == x);
  if (x != S::npos)
    assert(pos <= x && x + 1 <= s.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void test(const S& s, typename S::value_type c, typename S::size_type x) {
  LIBCPP_ASSERT_NOEXCEPT(s.find(c));
  assert(s.find(c) == x);
  if (x != S::npos)
    assert(0 <= x && x + 1 <= s.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(""), 'c', 0, S::npos);
  test(S(""), 'c', 1, S::npos);
  test(S("abcde"), 'c', 0, 2);
  test(S("abcde"), 'c', 1, 2);
  test(S("abcde"), 'c', 2, 2);
  test(S("abcde"), 'c', 4, S::npos);
  test(S("abcde"), 'c', 5, S::npos);
  test(S("abcde"), 'c', 6, S::npos);
  test(S("abcdeabcde"), 'c', 0, 2);
  test(S("abcdeabcde"), 'c', 1, 2);
  test(S("abcdeabcde"), 'c', 5, 7);
  test(S("abcdeabcde"), 'c', 9, S::npos);
  test(S("abcdeabcde"), 'c', 10, S::npos);
  test(S("abcdeabcde"), 'c', 11, S::npos);
  test(S("abcdeabcdeabcdeabcde"), 'c', 0, 2);
  test(S("abcdeabcdeabcdeabcde"), 'c', 1, 2);
  test(S("abcdeabcdeabcdeabcde"), 'c', 10, 12);
  test(S("abcdeabcdeabcdeabcde"), 'c', 19, S::npos);
  test(S("abcdeabcdeabcdeabcde"), 'c', 20, S::npos);
  test(S("abcdeabcdeabcdeabcde"), 'c', 21, S::npos);

  test(S(""), 'c', S::npos);
  test(S("abcde"), 'c', 2);
  test(S("abcdeabcde"), 'c', 2);
  test(S("abcdeabcdeabcdeabcde"), 'c', 2);
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
