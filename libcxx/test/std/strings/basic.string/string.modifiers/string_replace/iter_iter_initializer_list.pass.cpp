//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// UNSUPPORTED: c++03

// <string>

// basic_string& replace(const_iterator i1, const_iterator i2, initializer_list<charT> il); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool test() {
  {
    std::string s("123def456");
    s.replace(s.cbegin() + 3, s.cbegin() + 6, {'a', 'b', 'c'});
    assert(s == "123abc456");
  }
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    S s("123def456");
    s.replace(s.cbegin() + 3, s.cbegin() + 6, {'a', 'b', 'c'});
    assert(s == "123abc456");
  }

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
