//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// size_type rfind(charT c, size_type pos = npos) const; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(const S& s, typename S::value_type c, typename S::size_type pos, typename S::size_type x) {
  LIBCPP_ASSERT_NOEXCEPT(s.rfind(c, pos));
  assert(s.rfind(c, pos) == x);
  if (x != S::npos)
    assert(x <= pos && x + 1 <= s.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void test(const S& s, typename S::value_type c, typename S::size_type x) {
  LIBCPP_ASSERT_NOEXCEPT(s.rfind(c));
  assert(s.rfind(c) == x);
  if (x != S::npos)
    assert(x + 1 <= s.size());
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(""), 'b', 0, S::npos);
  test(S(""), 'b', 1, S::npos);
  test(S("abcde"), 'b', 0, S::npos);
  test(S("abcde"), 'b', 1, 1);
  test(S("abcde"), 'b', 2, 1);
  test(S("abcde"), 'b', 4, 1);
  test(S("abcde"), 'b', 5, 1);
  test(S("abcde"), 'b', 6, 1);
  test(S("abcdeabcde"), 'b', 0, S::npos);
  test(S("abcdeabcde"), 'b', 1, 1);
  test(S("abcdeabcde"), 'b', 5, 1);
  test(S("abcdeabcde"), 'b', 9, 6);
  test(S("abcdeabcde"), 'b', 10, 6);
  test(S("abcdeabcde"), 'b', 11, 6);
  test(S("abcdeabcdeabcdeabcde"), 'b', 0, S::npos);
  test(S("abcdeabcdeabcdeabcde"), 'b', 1, 1);
  test(S("abcdeabcdeabcdeabcde"), 'b', 10, 6);
  test(S("abcdeabcdeabcdeabcde"), 'b', 19, 16);
  test(S("abcdeabcdeabcdeabcde"), 'b', 20, 16);
  test(S("abcdeabcdeabcdeabcde"), 'b', 21, 16);

  test(S(""), 'b', S::npos);
  test(S("abcde"), 'b', 1);
  test(S("abcdeabcde"), 'b', 6);
  test(S("abcdeabcdeabcdeabcde"), 'b', 16);
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
