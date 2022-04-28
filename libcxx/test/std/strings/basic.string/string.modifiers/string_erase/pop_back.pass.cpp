//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <string>

// void pop_back(); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(S s, S expected)
{
    s.pop_back();
    LIBCPP_ASSERT(s.__invariants());
    assert(s[s.size()] == typename S::value_type());
    assert(s == expected);
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    test(S("abcde"), S("abcd"));
    test(S("abcdefghij"), S("abcdefghi"));
    test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrs"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S("abcde"), S("abcd"));
    test(S("abcdefghij"), S("abcdefghi"));
    test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrs"));
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
