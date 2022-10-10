//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// we get this comparison "for free" because the string implicitly converts to the string_view

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S, class SV>
TEST_CONSTEXPR_CXX20 void
test(const S& lhs, SV rhs, bool x)
{
    assert((lhs != rhs) == x);
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  typedef std::string_view SV;
  test(S(""), SV(""), false);
  test(S(""), SV("abcde"), true);
  test(S(""), SV("abcdefghij"), true);
  test(S(""), SV("abcdefghijklmnopqrst"), true);
  test(S("abcde"), SV(""), true);
  test(S("abcde"), SV("abcde"), false);
  test(S("abcde"), SV("abcdefghij"), true);
  test(S("abcde"), SV("abcdefghijklmnopqrst"), true);
  test(S("abcdefghij"), SV(""), true);
  test(S("abcdefghij"), SV("abcde"), true);
  test(S("abcdefghij"), SV("abcdefghij"), false);
  test(S("abcdefghij"), SV("abcdefghijklmnopqrst"), true);
  test(S("abcdefghijklmnopqrst"), SV(""), true);
  test(S("abcdefghijklmnopqrst"), SV("abcde"), true);
  test(S("abcdefghijklmnopqrst"), SV("abcdefghij"), true);
  test(S("abcdefghijklmnopqrst"), SV("abcdefghijklmnopqrst"), false);
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
#endif

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
