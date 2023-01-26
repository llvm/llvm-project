//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// const charT& front() const; // constexpr since C++20
//       charT& front(); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void
test(S s)
{
    const S& cs = s;
    ASSERT_SAME_TYPE(decltype( s.front()), typename S::reference);
    ASSERT_SAME_TYPE(decltype(cs.front()), typename S::const_reference);
    LIBCPP_ASSERT_NOEXCEPT(    s.front());
    LIBCPP_ASSERT_NOEXCEPT(   cs.front());
    assert(&cs.front() == &cs[0]);
    assert(&s.front() == &s[0]);
    s.front() = typename S::value_type('z');
    assert(s.front() == typename S::value_type('z'));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S("1"));
  test(S("1234567890123456789012345678901234567890"));
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
