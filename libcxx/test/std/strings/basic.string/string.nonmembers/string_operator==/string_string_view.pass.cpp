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
TEST_CONSTEXPR_CXX20 void test(const S& lhs, SV rhs, bool x) {
  assert((lhs == rhs) == x);
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    typedef std::string_view SV;
    test(S(""), SV(""), true);
    test(S(""), SV("abcde"), false);
    test(S(""), SV("abcdefghij"), false);
    test(S(""), SV("abcdefghijklmnopqrst"), false);
    test(S("abcde"), SV(""), false);
    test(S("abcde"), SV("abcde"), true);
    test(S("abcde"), SV("abcdefghij"), false);
    test(S("abcde"), SV("abcdefghijklmnopqrst"), false);
    test(S("abcdefghij"), SV(""), false);
    test(S("abcdefghij"), SV("abcde"), false);
    test(S("abcdefghij"), SV("abcdefghij"), true);
    test(S("abcdefghij"), SV("abcdefghijklmnopqrst"), false);
    test(S("abcdefghijklmnopqrst"), SV(""), false);
    test(S("abcdefghijklmnopqrst"), SV("abcde"), false);
    test(S("abcdefghijklmnopqrst"), SV("abcdefghij"), false);
    test(S("abcdefghijklmnopqrst"), SV("abcdefghijklmnopqrst"), true);
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    typedef std::basic_string_view<char, std::char_traits<char>> SV;
    test(S(""), SV(""), true);
    test(S(""), SV("abcde"), false);
    test(S(""), SV("abcdefghij"), false);
    test(S(""), SV("abcdefghijklmnopqrst"), false);
    test(S("abcde"), SV(""), false);
    test(S("abcde"), SV("abcde"), true);
    test(S("abcde"), SV("abcdefghij"), false);
    test(S("abcde"), SV("abcdefghijklmnopqrst"), false);
    test(S("abcdefghij"), SV(""), false);
    test(S("abcdefghij"), SV("abcde"), false);
    test(S("abcdefghij"), SV("abcdefghij"), true);
    test(S("abcdefghij"), SV("abcdefghijklmnopqrst"), false);
    test(S("abcdefghijklmnopqrst"), SV(""), false);
    test(S("abcdefghijklmnopqrst"), SV("abcde"), false);
    test(S("abcdefghijklmnopqrst"), SV("abcdefghij"), false);
    test(S("abcdefghijklmnopqrst"), SV("abcdefghijklmnopqrst"), true);
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
