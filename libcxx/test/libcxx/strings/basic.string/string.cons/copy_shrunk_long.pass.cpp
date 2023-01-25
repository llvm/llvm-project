//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(const basic_string<charT,traits,Allocator>& str);

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 bool test()
{
  // Tests that a long string holding a SSO size string results in
  // an SSO copy constructed value.
  S s1("1234567890123456789012345678901234567890123456789012345678901234567890");
  s1.resize(7);
  S s2(s1);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2 == s1);
  assert(s2.capacity() < sizeof(S));

  return true;
}

int main(int, char**)
{
  test<std::basic_string<char, std::char_traits<char>, test_allocator<char> > >();
#if TEST_STD_VER >= 11
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
#endif
#if TEST_STD_VER > 17
  static_assert(test<std::basic_string<char, std::char_traits<char>, test_allocator<char>>>());
  static_assert(test<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>());
#endif

  return 0;
}
