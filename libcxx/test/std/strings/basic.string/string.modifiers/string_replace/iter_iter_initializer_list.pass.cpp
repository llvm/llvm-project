//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string& replace(const_iterator i1, const_iterator i2, initializer_list<charT> il); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  S s("123def456");
  s.replace(s.cbegin() + 3, s.cbegin() + 6, {'a', 'b', 'c'});
  assert(s == "123abc456");
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
