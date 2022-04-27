//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <string>

// const charT& back() const; // constexpr since C++20
//       charT& back(); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(S s)
{
    const S& cs = s;
    ASSERT_SAME_TYPE(decltype( s.back()), typename S::reference);
    ASSERT_SAME_TYPE(decltype(cs.back()), typename S::const_reference);
    LIBCPP_ASSERT_NOEXCEPT(    s.back());
    LIBCPP_ASSERT_NOEXCEPT(   cs.back());
    assert(&cs.back() == &cs[cs.size()-1]);
    assert(&s.back() == &s[cs.size()-1]);
    s.back() = typename S::value_type('z');
    assert(s.back() == typename S::value_type('z'));
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::string S;
    test(S("1"));
    test(S("1234567890123456789012345678901234567890"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S("1"));
    test(S("1234567890123456789012345678901234567890"));
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
