//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// iterator insert(const_iterator p, charT c); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S& s, typename S::const_iterator p, typename S::value_type c, S expected) {
  LIBCPP_ASSERT(is_string_asan_correct(s));
  bool sufficient_cap             = s.size() < s.capacity();
  typename S::difference_type pos = p - s.begin();
  typename S::iterator i          = s.insert(p, c);
  LIBCPP_ASSERT(s.__invariants());
  assert(s == expected);
  assert(i - s.begin() == pos);
  assert(*i == c);
  if (sufficient_cap)
    assert(i == p);
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  S s;
  test(s, s.begin(), '1', S("1"));
  test(s, s.begin(), 'a', S("a1"));
  test(s, s.end(), 'b', S("a1b"));
  test(s, s.end() - 1, 'c', S("a1cb"));
  test(s, s.end() - 2, 'd', S("a1dcb"));
  test(s, s.end() - 3, '2', S("a12dcb"));
  test(s, s.end() - 4, '3', S("a132dcb"));
  test(s, s.end() - 5, '4', S("a1432dcb"));
  test(s, s.begin() + 1, '5', S("a51432dcb"));
  test(s, s.begin() + 2, '6', S("a561432dcb"));
  test(s, s.begin() + 3, '7', S("a5671432dcb"));
  test(s, s.begin() + 4, 'A', S("a567A1432dcb"));
  test(s, s.begin() + 5, 'B', S("a567AB1432dcb"));
  test(s, s.begin() + 6, 'C', S("a567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxxxxxxxxa567ABC1432dcb"));
  test(s, s.begin(), 'x', S("xxxxxxxxxxxxxxxxa567ABC1432dcb"));
  test(s, s.begin() + 1, 'x', S("xxxxxxxxxxxxxxxxxa567ABC1432dcb"));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char>>>();
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
