//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// const_reference at(size_type pos) const; // constexpr since C++20
//       reference at(size_type pos); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <cassert>

#include "min_allocator.h"

#include "make_string.h"
#include "test_macros.h"
#include "type_algorithms.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(S s, typename S::size_type pos)
{
    const S& cs = s;
    if (pos < s.size())
    {
        assert(s.at(pos) == s[pos]);
        assert(cs.at(pos) == cs[pos]);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    else if (!TEST_IS_CONSTANT_EVALUATED)
    {
        try
        {
            TEST_IGNORE_NODISCARD s.at(pos);
            assert(false);
        }
        catch (std::out_of_range&)
        {
            assert(pos >= s.size());
        }
        try
        {
            TEST_IGNORE_NODISCARD cs.at(pos);
            assert(false);
        }
        catch (std::out_of_range&)
        {
            assert(pos >= s.size());
        }
    }
#endif
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(), 0);
  test(S(MAKE_CSTRING(typename S::value_type, "123")), 0);
  test(S(MAKE_CSTRING(typename S::value_type, "123")), 1);
  test(S(MAKE_CSTRING(typename S::value_type, "123")), 2);
  test(S(MAKE_CSTRING(typename S::value_type, "123")), 3);
}

struct TestCaller {
  template <class T>
  TEST_CONSTEXPR_CXX20 void operator()() {
        test_string<std::basic_string<T> >();
#if TEST_STD_VER >= 11
        test_string<std::basic_string<T, std::char_traits<T>, min_allocator<T> > >();
#endif
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  meta::for_each(meta::character_types(), TestCaller());

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
