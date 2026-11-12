//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// iterator insert(const_iterator p, initializer_list<charT> il); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  S s("123456");
  typename S::iterator i = s.insert(s.begin() + 3, {'a', 'b', 'c'});
  assert(i - s.begin() == 3);
  assert(s == "123abc456");
  LIBCPP_ASSERT(is_string_asan_correct(s));
  typename S::iterator j = s.insert(s.begin() + 6, {'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
                                                    'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'});
  assert(j - s.begin() == 6);
  assert(s == "123abcaaaaaaaaaaaaaaaaaaaaaaaaa456");
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char> > >();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
