//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <string>

// void resize(size_type n); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(S s, typename S::size_type n, S expected)
{
    if (n <= s.max_size())
    {
        s.resize(n);
        LIBCPP_ASSERT(s.__invariants());
        assert(s == expected);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    else if (!TEST_IS_CONSTANT_EVALUATED)
    {
        try
        {
            s.resize(n);
            assert(false);
        }
        catch (std::length_error&)
        {
            assert(n > s.max_size());
        }
    }
#endif
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    test(S(), 0, S());
    test(S(), 1, S(1, '\0'));
    test(S(), 10, S(10, '\0'));
    test(S(), 100, S(100, '\0'));
    test(S("12345"), 0, S());
    test(S("12345"), 2, S("12"));
    test(S("12345"), 5, S("12345"));
    test(S("12345"), 15, S("12345\0\0\0\0\0\0\0\0\0\0", 15));
    test(S("12345678901234567890123456789012345678901234567890"), 0, S());
    test(S("12345678901234567890123456789012345678901234567890"), 10,
         S("1234567890"));
    test(S("12345678901234567890123456789012345678901234567890"), 50,
         S("12345678901234567890123456789012345678901234567890"));
    test(S("12345678901234567890123456789012345678901234567890"), 60,
         S("12345678901234567890123456789012345678901234567890\0\0\0\0\0\0\0\0\0\0", 60));
    test(S(), S::npos, S("not going to happen"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S(), 0, S());
    test(S(), 1, S(1, '\0'));
    test(S(), 10, S(10, '\0'));
    test(S(), 100, S(100, '\0'));
    test(S("12345"), 0, S());
    test(S("12345"), 2, S("12"));
    test(S("12345"), 5, S("12345"));
    test(S("12345"), 15, S("12345\0\0\0\0\0\0\0\0\0\0", 15));
    test(S("12345678901234567890123456789012345678901234567890"), 0, S());
    test(S("12345678901234567890123456789012345678901234567890"), 10,
         S("1234567890"));
    test(S("12345678901234567890123456789012345678901234567890"), 50,
         S("12345678901234567890123456789012345678901234567890"));
    test(S("12345678901234567890123456789012345678901234567890"), 60,
         S("12345678901234567890123456789012345678901234567890\0\0\0\0\0\0\0\0\0\0", 60));
    test(S(), S::npos, S("not going to happen"));
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
