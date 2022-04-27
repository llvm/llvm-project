//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <string>

// int compare(const basic_string& str) const // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 int sign(int x)
{
    if (x == 0)
        return 0;
    if (x < 0)
        return -1;
    return 1;
}

template <class S>
TEST_CONSTEXPR_CXX20 void
test(const S& s, const S& str, int x)
{
    LIBCPP_ASSERT_NOEXCEPT(s.compare(str));
    assert(sign(s.compare(str)) == sign(x));
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    test(S(""), S(""), 0);
    test(S(""), S("abcde"), -5);
    test(S(""), S("abcdefghij"), -10);
    test(S(""), S("abcdefghijklmnopqrst"), -20);
    test(S("abcde"), S(""), 5);
    test(S("abcde"), S("abcde"), 0);
    test(S("abcde"), S("abcdefghij"), -5);
    test(S("abcde"), S("abcdefghijklmnopqrst"), -15);
    test(S("abcdefghij"), S(""), 10);
    test(S("abcdefghij"), S("abcde"), 5);
    test(S("abcdefghij"), S("abcdefghij"), 0);
    test(S("abcdefghij"), S("abcdefghijklmnopqrst"), -10);
    test(S("abcdefghijklmnopqrst"), S(""), 20);
    test(S("abcdefghijklmnopqrst"), S("abcde"), 15);
    test(S("abcdefghijklmnopqrst"), S("abcdefghij"), 10);
    test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrst"), 0);
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S(""), S(""), 0);
    test(S(""), S("abcde"), -5);
    test(S(""), S("abcdefghij"), -10);
    test(S(""), S("abcdefghijklmnopqrst"), -20);
    test(S("abcde"), S(""), 5);
    test(S("abcde"), S("abcde"), 0);
    test(S("abcde"), S("abcdefghij"), -5);
    test(S("abcde"), S("abcdefghijklmnopqrst"), -15);
    test(S("abcdefghij"), S(""), 10);
    test(S("abcdefghij"), S("abcde"), 5);
    test(S("abcdefghij"), S("abcdefghij"), 0);
    test(S("abcdefghij"), S("abcdefghijklmnopqrst"), -10);
    test(S("abcdefghijklmnopqrst"), S(""), 20);
    test(S("abcdefghijklmnopqrst"), S("abcde"), 15);
    test(S("abcdefghijklmnopqrst"), S("abcdefghij"), 10);
    test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrst"), 0);
  }
#endif

#if TEST_STD_VER > 3
  { // LWG 2946
    std::string s = " !";
    assert(s.compare({"abc", 1}) < 0);
  }
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
