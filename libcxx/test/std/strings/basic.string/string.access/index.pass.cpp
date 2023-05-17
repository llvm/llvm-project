//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// const_reference operator[](size_type pos) const; // constexpr since C++20
//       reference operator[](size_type pos); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  S s("0123456789");
  const S& cs = s;
  ASSERT_SAME_TYPE(decltype( s[0]), typename S::reference);
  ASSERT_SAME_TYPE(decltype(cs[0]), typename S::const_reference);
  LIBCPP_ASSERT_NOEXCEPT(    s[0]);
  LIBCPP_ASSERT_NOEXCEPT(   cs[0]);
  for (typename S::size_type i = 0; i < cs.size(); ++i)
  {
      assert(s[i] == static_cast<char>('0' + i));
      assert(cs[i] == s[i]);
  }
  assert(cs[cs.size()] == '\0');
  const S s2 = S();
  assert(s2[0] == '\0');
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
